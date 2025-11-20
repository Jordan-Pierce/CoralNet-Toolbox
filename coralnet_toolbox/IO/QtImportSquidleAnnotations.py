import warnings
import os
import ujson as json

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QInputDialog)

from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportSquidleAnnotations:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def get_annotation_type_and_geometry(self, data, img_w, img_h):
        """
        Determines if the annotation is a Point, Box, or Polygon.
        Returns (type_string, geometry_object).
        """
        cx = data.get("point.x", 0)
        cy = data.get("point.y", 0)
        offsets = data.get("point.polygon", [])

        # Case 1: Point (No polygon offsets)
        if not offsets or len(offsets) == 0:
            # Convert normalized center to pixel point
            return "POINT", QPointF(cx * img_w, cy * img_h)

        # Convert all offsets to absolute pixel coordinates immediately
        # Abs = (Center + Offset) * Dimension
        pixel_points = []
        xs = []
        ys = []
        
        for off_x, off_y in offsets:
            abs_x = (cx + off_x) * img_w
            abs_y = (cy + off_y) * img_h
            pixel_points.append(QPointF(abs_x, abs_y))
            xs.append(abs_x)
            ys.append(abs_y)

        num_vertices = len(pixel_points)

        # Case 2: Polygon (Explicitly defined by vertex count heuristics)
        # If > 5 vertices, it's definitely a polygon. 
        # If 4 vertices (unclosed loop), it's a polygon/quad.
        if num_vertices > 5 or num_vertices == 4:
            return "POLYGON", pixel_points

        # Case 3: Ambiguous (5 vertices, usually a closed loop)
        # We check for Bounding Box properties (2 unique X and 2 unique Y)
        if num_vertices == 5:
            # Use a small epsilon for float comparison if needed, 
            # but rounding to int for unique counting is usually safe for pixel coords
            unique_x = len(set([round(x, 2) for x in xs]))
            unique_y = len(set([round(y, 2) for y in ys]))

            if unique_x == 2 and unique_y == 2:
                # It is an Axis-Aligned Bounding Box
                # Return QRectF with topLeft and bottomRight points
                return "RECTANGLE", QRectF(QPointF(min(xs), min(ys)), QPointF(max(xs), max(ys)))
            else:
                # It is a 5-point Polygon (e.g. a pentagon or rotated box)
                return "POLYGON", pixel_points

        # Fallback for triangles (3 points)
        return "POLYGON", pixel_points

    def import_annotations(self):
        """Import annotations from Squidle+ JSON files."""
        self.main_window.untoggle_all_tools()

        if not self.annotation_window.active_image:
            QMessageBox.warning(self.annotation_window,
                                "No Images Loaded",
                                "Please load images first before importing annotations.")
            return

        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self.annotation_window,
                                                     "Import Squidle+ Annotations",
                                                     "",
                                                     "JSON Files (*.json);;All Files (*)",
                                                     options=options)

        if not file_paths:
            return

        # Ask for patch size (fallback for Point annotations)
        patch_size, ok = QInputDialog.getInt(self.annotation_window,
                                             "Patch Annotation Size",
                                             "Enter the default size for Point annotations:",
                                             224, 1, 10000, 1)
        if not ok:
            return

        try:
            all_records = []
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    # Squidle export might be a list of objects, or a paginated result dict
                    data = json.load(file)
                    if isinstance(data, list):
                        all_records.extend(data)
                    elif isinstance(data, dict) and "objects" in data:
                        # Handle API response format
                        all_records.extend(data["objects"])
                    elif isinstance(data, dict):
                        all_records.append(data)

            # Map image names to full local paths
            # Squidle "point.media.key" usually corresponds to the filename
            image_path_map = {os.path.basename(path): path for path in self.image_window.raster_manager.image_paths}

            total_ops = len(all_records)

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Reading File",
                                f"An error occurred while reading the file: {str(e)}")
            return

        # UI Setup
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.annotation_window, title="Importing Squidle+ Annotations")
        progress_bar.show()
        progress_bar.start_progress(total_ops)

        images_to_update = set()
        skipped_count = 0

        try:
            # PHASE 1: Import all labels first (batch)
            unique_labels = {}
            for record in all_records:
                label_code = record.get("label.name", "Review")
                if label_code not in unique_labels:
                    label_obj = self.label_window.add_label_if_not_exists(
                        label_code,
                        label_code,
                        None,
                        None
                    )
                    if label_obj:
                        unique_labels[label_code] = label_obj
            
            # PHASE 2: Create all annotation objects in memory first
            new_annotations = []
            for record in all_records:
                try:
                    # 1. Link to Image
                    media_key = record.get("point.media.key") or record.get("media_path", "")
                    image_basename = os.path.basename(media_key)
                    
                    # If exact match fails, try matching without extension or loose matching could go here
                    image_path = image_path_map.get(image_basename)
                    
                    if not image_path:
                        # Try to find if the record name is inside any of the loaded images keys
                        # This handles cases where squidle path is "foo/bar/img.jpg" and we just have "img.jpg"
                        for k, v in image_path_map.items():
                            if k in media_key:
                                image_path = v
                                break
                    
                    if not image_path:
                        skipped_count += 1
                        progress_bar.update_progress()
                        continue

                    images_to_update.add(image_path)

                    # 2. Get label (already created in phase 1)
                    label_code = record.get("label.name", "Review")
                    label_obj = unique_labels.get(label_code)
                    
                    if not label_obj:
                        skipped_count += 1
                        progress_bar.update_progress()
                        continue

                    # 3. Determine Geometry & Convert Coordinates
                    # We need image dimensions to convert normalized -> pixel
                    raster = self.image_window.raster_manager.get_raster(image_path)
                    if not raster or not hasattr(raster, 'width') or not hasattr(raster, 'height'):
                        skipped_count += 1 
                        progress_bar.update_progress()
                        continue
                    
                    img_w = raster.width
                    img_h = raster.height
                    
                    # Validate image dimensions are valid
                    if img_w <= 0 or img_h <= 0:
                        skipped_count += 1
                        progress_bar.update_progress()
                        continue

                    anno_type, geometry = self.get_annotation_type_and_geometry(record, img_w, img_h)

                    # 4. Create Annotation Object
                    annotation = None

                    if anno_type == "POINT":
                        center_xy = geometry
                        annotation = PatchAnnotation(
                            center_xy=center_xy,  # QPointF center
                            annotation_size=patch_size,
                            short_label_code=label_obj.short_label_code,
                            long_label_code=label_obj.long_label_code,
                            color=label_obj.color,
                            image_path=image_path,
                            label_id=label_obj.id
                        )
                    elif anno_type == "RECTANGLE":
                        top_left = geometry.topLeft()
                        bottom_right = geometry.bottomRight()
                        annotation = RectangleAnnotation(
                            top_left=top_left,  # QPointF
                            bottom_right=bottom_right,  # QPointF
                            short_label_code=label_obj.short_label_code,
                            long_label_code=label_obj.long_label_code,
                            color=label_obj.color,
                            image_path=image_path,
                            label_id=label_obj.id
                        )
                    elif anno_type == "POLYGON":
                        points = geometry  # List of QPointF
                        annotation = PolygonAnnotation(
                            points=points,
                            short_label_code=label_obj.short_label_code,
                            long_label_code=label_obj.long_label_code,
                            color=label_obj.color,
                            image_path=image_path,
                            label_id=label_obj.id
                        )

                    # 5. Metadata Stashing
                    if annotation:
                        # Handle Verification status
                        # Squidle 'needs_review' -> True usually implies NOT verified.
                        needs_review = record.get("needs_review", False)
                        annotation.set_verified(not needs_review)

                        # Filter and store relevant Squidle metadata
                        # Only keep essential fields to avoid serialization issues
                        annotation.data = record.copy()
                        annotation.data['point.polygon']  = []  # Remove polygon offsets to reduce size

                        new_annotations.append(annotation)

                except Exception as inner_e:
                    import traceback
                    print(f"Skipping record due to error: {inner_e}")
                    traceback.print_exc()
                    skipped_count += 1
                
                progress_bar.update_progress()

            # PHASE 3: Batch Add to Project
            if new_annotations:
                self.annotation_window.add_annotations(new_annotations)

            # PHASE 4: Update UI
            for path in images_to_update:
                self.image_window.update_image_annotations(path)

            self.annotation_window.load_annotations()

            msg = f"Successfully imported {len(new_annotations)} annotations."
            if skipped_count > 0:
                msg += f"\n\nSkipped {skipped_count} records (images not found)."
            
            QMessageBox.information(self.annotation_window, "Import Complete", msg)

        except Exception as e:
            QMessageBox.critical(self.annotation_window,
                                 "Critical Error",
                                 f"Failed during import process: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()