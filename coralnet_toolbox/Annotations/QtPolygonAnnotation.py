import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import math

from rasterio.windows import Window

from shapely.ops import split, unary_union
from shapely.geometry import Point, Polygon, LineString

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPathItem
from PyQt5.QtGui import (QPixmap, QColor, QPen, QBrush, QPolygonF,
                         QPainter, QRegion, QImage, QPainterPath)

from coralnet_toolbox.Annotations.QtAnnotation import Annotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

from coralnet_toolbox.utilities import densify_polygon
from coralnet_toolbox.utilities import simplify_polygon
from coralnet_toolbox.utilities import rasterio_to_cropped_image


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PolygonAnnotation(Annotation):
    def __init__(self,
                 points: list,
                 short_label_code: str,
                 long_label_code: str,
                 color: QColor,
                 image_path: str,
                 label_id: str,
                 transparency: int = 128,
                 show_msg: bool = False,
                 holes: list = None):
        super().__init__(short_label_code, long_label_code, color, image_path, label_id, transparency, show_msg)

        self.center_xy = QPointF(0, 0)
        self.cropped_bbox = (0, 0, 0, 0)
        self.annotation_size = 0
        
        # Initialize holes, ensuring it's always a list
        self.holes = holes if holes is not None else []

        # Set the main polygon points and calculate initial properties
        self.set_precision(points, True)
        self.set_centroid()
        self.set_cropped_bbox()

    def set_precision(self, points: list, reduce: bool = True):
        """
        Set the precision of the outer points and all inner holes.

        Args:
            points: List of QPointF vertices for the outer boundary.
            reduce: Whether to round coordinates to a set number of decimal places.
        """
        # Process and assign the outer boundary points
        if reduce:
            self.points = [QPointF(round(p.x(), 6), round(p.y(), 6)) for p in points]
        else:
            self.points = points

        # Process each list of points for the inner holes, if any
        if self.holes and reduce:
            processed_holes = []
            for hole in self.holes:
                processed_hole = [QPointF(round(p.x(), 6), round(p.y(), 6)) for p in hole]
                processed_holes.append(processed_hole)
            self.holes = processed_holes

    def set_centroid(self):
        """
        Calculate the true geometric centroid of the polygon, accounting for any holes.
        """
        try:
            # Simple average of the outer points for stability
            centroid_x = sum(point.x() for point in self.points) / len(self.points)
            centroid_y = sum(point.y() for point in self.points) / len(self.points)
            self.center_xy = QPointF(centroid_x, centroid_y)
            
        except Exception:
            # Convert the QPointF lists to coordinate tuples for Shapely
            shell_coords = [(p.x(), p.y()) for p in self.points]
            holes_coords = [[(p.x(), p.y()) for p in hole] for hole in self.holes]

            # Create a Shapely polygon with its shell and holes
            shapely_polygon = Polygon(shell=shell_coords, holes=holes_coords)

            # Get the true centroid from the Shapely object
            centroid = shapely_polygon.centroid

            # Update the annotation's center_xy with the new coordinates
            self.center_xy = QPointF(centroid.x, centroid.y)

    def set_cropped_bbox(self):
        """Calculate the bounding box of the polygon defined by the points."""
        min_x = min(point.x() for point in self.points)
        min_y = min(point.y() for point in self.points)
        max_x = max(point.x() for point in self.points)
        max_y = max(point.y() for point in self.points)
        self.cropped_bbox = (min_x, min_y, max_x, max_y)
        self.annotation_size = int(max(max_x - min_x, max_y - min_y))

    def contains_point(self, point: QPointF) -> bool:
        """
        Check if the given point is inside the polygon, excluding any holes.
        """
        try:
            # Convert the QPointF lists to coordinate tuples for Shapely
            shell_coords = [(p.x(), p.y()) for p in self.points]
            holes_coords = [[(p.x(), p.y()) for p in hole] for hole in self.holes]

            # Create a Shapely polygon with its shell and holes
            shapely_polygon = Polygon(shell=shell_coords, holes=holes_coords)

            # Convert the input QPointF to a Shapely Point
            shapely_point = Point(point.x(), point.y())

            # Return Shapely's boolean result for the containment check
            return shapely_polygon.contains(shapely_point)

        except Exception:
            # If Shapely fails, fall back to the original implementation which
            # checks against the outer boundary only.
            polygon = QPolygonF(self.points)
            return polygon.containsPoint(point, Qt.OddEvenFill)

    def get_centroid(self):
        """Get the centroid of the annotation."""
        return (float(self.center_xy.x()), float(self.center_xy.y()))

    def get_area(self):
        """
        Calculate the net area of the polygon (outer area - inner holes' area).
        """
        # A shape with fewer than 3 points has no area.
        if len(self.points) < 3:
            return 0.0

        try:
            # Convert the QPointF lists to coordinate tuples for Shapely
            shell_coords = [(p.x(), p.y()) for p in self.points]
            holes_coords = [[(p.x(), p.y()) for p in hole] for hole in self.holes]

            # Create a Shapely polygon with its shell and holes
            shapely_polygon = Polygon(shell=shell_coords, holes=holes_coords)

            # Return the net area calculated by Shapely
            return shapely_polygon.area

        except Exception:
            # If Shapely fails (e.g., due to invalid geometry), fall back to
            # calculating the gross area of the outer polygon using the shoelace formula.
            # This is the original implementation.
            area = 0.0
            n = len(self.points)
            for i in range(n):
                j = (i + 1) % n
                area += self.points[i].x() * self.points[j].y()
                area -= self.points[j].x() * self.points[i].y()
            return abs(area) / 2.0

    def get_perimeter(self):
        """
        Calculate the total perimeter of the polygon (outer boundary + all hole boundaries).
        """
        # A shape with fewer than 2 points has no length.
        if len(self.points) < 2:
            return 0.0

        try:
            # Convert the QPointF lists to coordinate tuples for Shapely
            shell_coords = [(p.x(), p.y()) for p in self.points]
            holes_coords = [[(p.x(), p.y()) for p in hole] for hole in self.holes]

            # Create a Shapely polygon with its shell and holes
            shapely_polygon = Polygon(shell=shell_coords, holes=holes_coords)

            # Return the total perimeter (length) calculated by Shapely
            return shapely_polygon.length

        except Exception:
            # If Shapely fails, fall back to calculating the perimeter of the
            # outer boundary only. This is the original implementation.
            perimeter = 0.0
            n = len(self.points)
            for i in range(n):
                j = (i + 1) % n
                dx = self.points[i].x() - self.points[j].x()
                dy = self.points[i].y() - self.points[j].y()
                distance = math.sqrt(dx * dx + dy * dy)
                perimeter += distance
            return perimeter

    def get_polygon(self):
        """Get the polygon representation of this polygon annotation."""
        return QPolygonF(self.points)
    
    def get_painter_path(self) -> QPainterPath:
        """
        Get a QPainterPath representation of the annotation, including holes.

        This is the correct object to use for rendering complex polygons.
        """
        path = QPainterPath()

        # 1. Add the outer boundary to the path
        path.addPolygon(QPolygonF(self.points))

        # 2. Add each of the inner holes to the path
        for hole in self.holes:
            path.addPolygon(QPolygonF(hole))

        # 3. Set the fill rule, which tells the painter to treat
        #    overlapping polygons as holes.
        path.setFillRule(Qt.OddEvenFill)

        return path

    def get_bounding_box_top_left(self):
        """Get the top-left corner of the annotation's bounding box."""
        return QPointF(self.cropped_bbox[0], self.cropped_bbox[1])

    def get_bounding_box_bottom_right(self):
        """Get the bottom-right corner of the annotation's bounding box."""
        return QPointF(self.cropped_bbox[2], self.cropped_bbox[3])

    def get_cropped_image_graphic(self):
        """
        Get the cropped image with the polygon and its holes correctly masked.
        """
        if self.cropped_image is None:
            return None

        # --- Create the painter path with translated coordinates ---
        # The path needs coordinates relative to the cropped image's top-left corner.
        offset_x = self.cropped_bbox[0]
        offset_y = self.cropped_bbox[1]

        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)

        # Add the translated outer boundary
        outer_boundary = QPolygonF([QPointF(p.x() - offset_x, p.y() - offset_y) for p in self.points])
        path.addPolygon(outer_boundary)

        # Add the translated holes
        for hole in self.holes:
            inner_boundary = QPolygonF([QPointF(p.x() - offset_x, p.y() - offset_y) for p in hole])
            path.addPolygon(inner_boundary)
        # ---------------------------------------------------------

        # Create a QImage for the mask with a transparent background
        masked_image = QImage(self.cropped_image.size(), QImage.Format_ARGB32)
        masked_image.fill(Qt.transparent)

        # Create a painter to draw the path onto the mask
        mask_painter = QPainter(masked_image)
        mask_painter.setRenderHint(QPainter.Antialiasing)
        mask_painter.setBrush(QBrush(Qt.white))  # White fill for the mask area
        mask_painter.setPen(Qt.NoPen)
        mask_painter.drawPath(path)  # Use drawPath instead of drawPolygon
        mask_painter.end()

        # Convert the mask to a QRegion for clipping
        mask_pixmap = QPixmap.fromImage(masked_image)
        mask_bitmap = mask_pixmap.createMaskFromColor(Qt.white, Qt.MaskOutColor)
        mask_region = QRegion(mask_bitmap)

        # --- Compose the final graphic ---
        cropped_image_graphic = QPixmap(self.cropped_image.size())
        result_painter = QPainter(cropped_image_graphic)
        result_painter.setRenderHint(QPainter.Antialiasing)

        # Draw the full original image at 50% opacity
        result_painter.setOpacity(0.5)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw the full-opacity image inside the masked region (the annotation area)
        result_painter.setOpacity(1.0)
        result_painter.setClipRegion(mask_region)
        result_painter.drawPixmap(0, 0, self.cropped_image)

        # Draw the outline of the path (outer and inner boundaries)
        pen = QPen(Qt.black)
        pen.setStyle(Qt.SolidLine)
        pen.setWidth(1)
        result_painter.setPen(pen)
        result_painter.setClipping(False)  # Disable clipping for the outline
        result_painter.drawPath(path)  # Use drawPath for the outline as well
        result_painter.end()

        return cropped_image_graphic

    def create_cropped_image(self, rasterio_src):
        """Create a cropped image from the rasterio source based on the polygon points."""
        # Set the rasterio source for the annotation
        self.rasterio_src = rasterio_src
        # Set the cropped bounding box for the annotation
        self.set_cropped_bbox()
        # Get the bounding box of the polygon
        min_x, min_y, max_x, max_y = self.cropped_bbox

        # Calculate the window for rasterio
        window = Window(
            col_off=max(0, int(min_x)),
            row_off=max(0, int(min_y)),
            width=min(rasterio_src.width - int(min_x), int(max_x - min_x)),
            height=min(rasterio_src.height - int(min_y), int(max_y - min_y))
        )

        # Convert rasterio to QImage
        q_image = rasterio_to_cropped_image(self.rasterio_src, window)
        # Convert QImage to QPixmap
        self.cropped_image = QPixmap.fromImage(q_image)

        self.annotationUpdated.emit(self)  # Notify update

    def create_graphics_item(self, scene: QGraphicsScene):
        """
        Create all graphics items for the annotation and add them to the scene.
        
        This now uses QGraphicsPathItem to correctly render holes.
        """
        # Get the complete shape (with holes) as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem, the correct item for a QPainterPath.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle grouping, styling, and adding to the scene.
        super().create_graphics_item(scene)

    def update_graphics_item(self):
        """
        Update the graphical representation of the polygon annotation.

        This now uses QGraphicsPathItem to correctly re-render holes.
        """
        # Get the complete shape (with holes) as a QPainterPath.
        path = self.get_painter_path()
        
        # Use a QGraphicsPathItem to correctly represent the shape.
        self.graphics_item = QGraphicsPathItem(path)
        
        # Call the parent class method to handle rebuilding the graphics group.
        super().update_graphics_item()
    
    def update_polygon(self, delta):
        """
        Simplify or densify the polygon and its holes based on wheel movement.
        """
        # Determine which function to use based on the delta
        if delta < 0:
            # Simplify: increase tolerance (less detail)
            self.tolerance = min(self.tolerance + 0.05, 2.0)
            process_function = lambda pts: simplify_polygon(pts, self.tolerance)
        elif delta > 0:
            # Densify: decrease segment length (more detail)
            process_function = densify_polygon
        else:
            # No change
            return

        # --- Process the Outer Boundary ---
        xy_points = [(p.x(), p.y()) for p in self.points]
        updated_coords = process_function(xy_points)
        
        # --- Process Each of the Inner Holes ---
        updated_holes = []
        if self.holes:
            for hole in self.holes:
                xy_hole_points = [(p.x(), p.y()) for p in hole]
                updated_hole_coords = process_function(xy_hole_points)
                updated_holes.append([QPointF(x, y) for x, y in updated_hole_coords])
        
        # Update the holes attribute before calling set_precision
        self.holes = updated_holes

        # --- Finalize and Update ---
        # Convert outer boundary points and set precision for all points
        final_points = [QPointF(x, y) for x, y in updated_coords]
        self.set_precision(final_points)

        # Recalculate properties and refresh the graphics
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def update_location(self, new_center_xy: QPointF):
        """
        Update the location of the annotation by moving it to a new center point.
        This now moves the outer boundary and all holes together.
        """
        # Clear the machine confidence
        self.update_user_confidence(self.label)

        # Calculate the distance to move (delta)
        delta = QPointF(round(new_center_xy.x() - self.center_xy.x(), 2),
                        round(new_center_xy.y() - self.center_xy.y(), 2))

        # Move the outer boundary points
        new_points = [point + delta for point in self.points]

        # Move all points for each hole
        new_holes = []
        for hole in self.holes:
            moved_hole = [point + delta for point in hole]
            new_holes.append(moved_hole)
        self.holes = new_holes

        # Update precision, recalculate properties, and refresh the graphics
        self.set_precision(new_points)
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)  # Notify update

    def update_annotation_size(self, delta: float):
        """
        Grow/shrink the polygon and its holes by scaling vertices radially from the centroid.
        """
        self.update_user_confidence(self.label)

        if len(self.points) < 3:
            return

        # 1. Use the true geometric centroid as the pivot for scaling.
        # This is correctly calculated by the new set_centroid() method.
        centroid_x = self.center_xy.x()
        centroid_y = self.center_xy.y()

        # 2. Determine the scale factor (this logic remains the same).
        step = 0.01  # Adjust for finer or coarser changes
        if delta > 1.0:
            scale = 1.0 + step
        elif delta < 1.0:
            scale = 1.0 - step
        else:
            scale = 1.0

        # 3. Scale the outer boundary points.
        new_points = []
        for p in self.points:
            dx = p.x() - centroid_x
            dy = p.y() - centroid_y
            new_x = centroid_x + dx * scale
            new_y = centroid_y + dy * scale
            new_points.append(QPointF(new_x, new_y))
            
        # 4. Scale all points for each hole using the same logic.
        new_holes = []
        for hole in self.holes:
            scaled_hole = []
            for p in hole:
                dx = p.x() - centroid_x
                dy = p.y() - centroid_y
                new_x = centroid_x + dx * scale
                new_y = centroid_y + dy * scale
                scaled_hole.append(QPointF(new_x, new_y))
            new_holes.append(scaled_hole)
        self.holes = new_holes

        # 5. Update precision, recalculate properties, and refresh the graphics.
        self.set_precision(new_points)
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    def resize(self, handle: str, new_pos: QPointF):
        """
        Resize the annotation by moving a specific handle (vertex) to a new position.
        The handle format is updated to support holes: 'point_{poly_index}_{vertex_index}'.
        """
        self.update_user_confidence(self.label)

        if not handle.startswith("point_"):
            return

        try:
            # Parse the new handle format: "point_outer_5" or "point_0_2"
            _, poly_index_str, vertex_index_str = handle.split("_")
            vertex_index = int(vertex_index_str)

            # --- Modify the correct list of points ---
            if poly_index_str == "outer":
                # Handle resizing the outer boundary
                if 0 <= vertex_index < len(self.points):
                    new_points = self.points.copy()
                    new_points[vertex_index] = new_pos
                    # set_precision will handle updating self.points
                    self.set_precision(new_points)
            else:
                # Handle resizing one of the holes
                poly_index = int(poly_index_str)
                if 0 <= poly_index < len(self.holes):
                    if 0 <= vertex_index < len(self.holes[poly_index]):
                        # Create a copy, modify it, and update the list of holes
                        new_hole = self.holes[poly_index].copy()
                        new_hole[vertex_index] = new_pos
                        self.holes[poly_index] = new_hole
                        # set_precision will handle the holes list in-place
                        self.set_precision(self.points)

        except (ValueError, IndexError):
            # Fail gracefully if the handle format is invalid
            return

        # --- Recalculate properties and refresh the graphics ---
        self.set_centroid()
        self.set_cropped_bbox()
        self.update_graphics_item()
        self.annotationUpdated.emit(self)

    @classmethod
    def combine(cls, annotations: list):
        """
        Combine annotations using Shapely's union operation.
        Returns a single PolygonAnnotation (if merged) or a MultiPolygonAnnotation (if disjoint).
        """
        if not annotations:
            return None
        if len(annotations) == 1:
            return annotations[0]

        try:
            # 1. Convert all input annotations to Shapely Polygons, preserving their holes.
            shapely_polygons = []
            for anno in annotations:
                shell = [(p.x(), p.y()) for p in anno.points]
                holes = [[(p.x(), p.y()) for p in hole] for hole in getattr(anno, 'holes', [])]
                shapely_polygons.append(Polygon(shell, holes))

            # 2. Perform the union operation.
            merged_geom = unary_union(shapely_polygons)
            
            # --- Get properties from the first annotation to transfer to the new one ---
            first_anno = annotations[0]
            common_args = {
                "short_label_code": first_anno.label.short_label_code,
                "long_label_code": first_anno.label.long_label_code,
                "color": first_anno.label.color,
                "image_path": first_anno.image_path,
                "label_id": first_anno.label.id
            }

            # 3. Check the result and build the appropriate new annotation.
            if merged_geom.geom_type == 'Polygon':
                # The result is a single polygon (potentially with new holes).
                exterior_points = [QPointF(x, y) for x, y in merged_geom.exterior.coords]
                interior_holes = [[QPointF(x, y) for x, y in interior.coords] for interior in merged_geom.interiors]
                
                return cls(points=exterior_points, holes=interior_holes, **common_args)

            elif merged_geom.geom_type == 'MultiPolygon':
                # The result is multiple, disjoint polygons. Create a MultiPolygonAnnotation.
                new_polygons = []
                for poly in merged_geom.geoms:
                    exterior_points = [QPointF(x, y) for x, y in poly.exterior.coords]
                    interior_holes = [[QPointF(x, y) for x, y in interior.coords] for interior in poly.interiors]
                    new_polygons.append(cls(points=exterior_points, holes=interior_holes, **common_args))
                
                return MultiPolygonAnnotation(polygons=new_polygons, **common_args)
            
            else:
                # The geometry is empty or an unexpected type.
                return None

        except Exception as e:
            print(f"Error during polygon combination: {e}")
            return None

    @classmethod
    def cut(cls, annotation, cutting_points: list):
        """
        Cut a polygon annotation where it intersects with a cutting line.
        This now correctly handles cutting polygons that contain holes.
        """
        if not annotation or not cutting_points or len(cutting_points) < 2:
            return [annotation] if annotation else []

        # 1. Create a Shapely Polygon, including its shell and any holes.
        try:
            shell_coords = [(p.x(), p.y()) for p in annotation.points]
            if len(shell_coords) < 3:  # Not a valid polygon to cut
                return [annotation]
            holes_coords = [[(p.x(), p.y()) for p in hole] for hole in getattr(annotation, 'holes', [])]
            polygon = Polygon(shell_coords, holes_coords)
            
        except Exception:
            # Invalid geometry to begin with
            return [annotation]

        # Create the cutting line
        line_points = [(point.x(), point.y()) for point in cutting_points]
        cutting_line = LineString(line_points)

        # Check if the line intersects with the polygon
        if not polygon.intersects(cutting_line):
            return [annotation]  # No intersection, return original

        try:
            # 2. Split the polygon; Shapely handles the holes automatically.
            split_geometries = split(polygon, cutting_line)

            result_annotations = []
            min_area = 10  # Minimum area threshold to avoid tiny fragments

            # 3. Reconstruct new PolygonAnnotations from the resulting geometries.
            for geom in split_geometries.geoms:
                if geom.area < min_area or not isinstance(geom, Polygon):
                    continue

                # Extract the exterior coordinates
                new_points = [QPointF(x, y) for x, y in geom.exterior.coords]
                # Also extract the coordinates for any new holes
                new_holes = [[QPointF(x, y) for x, y in interior.coords] for interior in geom.interiors]

                if len(new_points) < 3:
                    continue
                
                # Create a new annotation with the new points and holes
                new_anno = cls(
                    points=new_points,
                    holes=new_holes,  # Pass the new holes
                    short_label_code=annotation.label.short_label_code,
                    long_label_code=annotation.label.long_label_code,
                    color=annotation.label.color,
                    image_path=annotation.image_path,
                    label_id=annotation.label.id
                )

                # Transfer rasterio source if available
                if hasattr(annotation, 'rasterio_src') and annotation.rasterio_src is not None:
                    new_anno.rasterio_src = annotation.rasterio_src
                    new_anno.create_cropped_image(new_anno.rasterio_src)

                result_annotations.append(new_anno)

            return result_annotations if result_annotations else [annotation]

        except Exception as e:
            print(f"Error during polygon cutting: {e}")
            return [annotation]
        
    @classmethod
    def subtract(cls, base_annotation, cutter_annotations: list):
        """
        Performs a symmetrical subtraction.

        Subtracts the combined area of cutter_annotations from the base_annotation,
        and also subtracts the base_annotation from each of the cutter_annotations.
        Returns a list of all resulting annotation fragments.
        """
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation

        def _create_annotations_from_geom(geom, source_annotation):
            """Creates appropriate Annotation objects from a Shapely geometry."""
            if geom.is_empty:
                return []

            common_args = {
                "short_label_code": source_annotation.label.short_label_code,
                "long_label_code": source_annotation.label.long_label_code,
                "color": source_annotation.label.color,
                "image_path": source_annotation.image_path,
                "label_id": source_annotation.label.id
            }

            if geom.geom_type == 'Polygon':
                exterior_points = [QPointF(x, y) for x, y in geom.exterior.coords]
                interior_holes = [[QPointF(x, y) for x, y in interior.coords] for interior in geom.interiors]
                return [cls(points=exterior_points, holes=interior_holes, **common_args)]

            elif geom.geom_type == 'MultiPolygon':
                new_polygons = []
                for poly in geom.geoms:
                    if poly.is_empty: continue
                    exterior_points = [QPointF(x, y) for x, y in poly.exterior.coords]
                    interior_holes = [[QPointF(x, y) for x, y in interior.coords] for interior in poly.interiors]
                    new_polygons.append(cls(points=exterior_points, holes=interior_holes, **common_args))
                
                if new_polygons:
                    return [MultiPolygonAnnotation(polygons=new_polygons, **common_args)]

            return []

        if not base_annotation or not cutter_annotations:
            return []

        try:
            # --- Convert all annotations to Shapely objects ---
            base_shell = [(p.x(), p.y()) for p in base_annotation.points]
            base_holes = [[(p.x(), p.y()) for p in hole] for hole in getattr(base_annotation, 'holes', [])]
            base_polygon = Polygon(base_shell, base_holes)

            cutter_polygons, cutter_source_annotations = [], []
            for anno in cutter_annotations:
                if isinstance(anno, MultiPolygonAnnotation):
                    for poly in anno.polygons:
                        shell = [(p.x(), p.y()) for p in poly.points]
                        holes = [[(p.x(), p.y()) for p in hole] for hole in getattr(poly, 'holes', [])]
                        cutter_polygons.append(Polygon(shell, holes))
                        cutter_source_annotations.append(poly)
                else:
                    shell = [(p.x(), p.y()) for p in anno.points]
                    holes = [[(p.x(), p.y()) for p in hole] for hole in getattr(anno, 'holes', [])]
                    cutter_polygons.append(Polygon(shell, holes))
                    cutter_source_annotations.append(anno)
            
            cutter_union = unary_union(cutter_polygons)

            if not base_polygon.intersects(cutter_union):
                return []  # No overlap, so return an empty list to signal no-op

            all_results = []

            # --- 1. Calculate Base - CutterUnion ---
            result_base_geom = base_polygon.difference(cutter_union)
            all_results.extend(_create_annotations_from_geom(result_base_geom, base_annotation))

            # --- 2. Calculate each Cutter - Base ---
            for i, cutter_poly in enumerate(cutter_polygons):
                source_anno = cutter_source_annotations[i]
                result_cutter_geom = cutter_poly.difference(base_polygon)
                all_results.extend(_create_annotations_from_geom(result_cutter_geom, source_anno))

            return all_results

        except Exception as e:
            print(f"Error during polygon subtraction: {e}")
            return []
        
    def to_dict(self):
        """Convert the annotation to a dictionary, including points and holes."""
        base_dict = super().to_dict()
        base_dict.update({
            'points': [(point.x(), point.y()) for point in self.points],
            'holes': [[(p.x(), p.y()) for p in hole] for hole in self.holes]
        })
        return base_dict

    @classmethod
    def from_dict(cls, data, label_window):
        """Create a PolygonAnnotation object from a dictionary, including holes."""
        points = [QPointF(x, y) for x, y in data['points']]
        
        # Check for and process hole data if it exists.
        holes_data = data.get('holes', [])
        holes = [[QPointF(x, y) for x, y in hole_data] for hole_data in holes_data]

        # Pass the points and holes to the constructor.
        annotation = cls(
            points=points,
            holes=holes,
            short_label_code=data['label_short_code'],
            long_label_code=data['label_long_code'],
            color=QColor(*data['annotation_color']),
            image_path=data['image_path'],
            label_id=data['label_id']
        )
        annotation.data = data.get('data', {})

        # --- Remainder of the method is for handling confidence scores ---
        machine_confidence = {}
        for short_label_code, confidence in data.get('machine_confidence', {}).items():
            label = label_window.get_label_by_short_code(short_label_code)
            if label:
                machine_confidence[label] = confidence

        annotation.update_machine_confidence(machine_confidence, from_import=True)

        if 'verified' in data:
            annotation.set_verified(data['verified'])

        return annotation

    def __repr__(self):
        """Return a string representation of the PolygonAnnotation object."""
        return (f"PolygonAnnotation(id={self.id}, points={self.points}, "
                f"annotation_color={self.label.color.name()}, "
                f"image_path={self.image_path}, "
                f"label={self.label.short_label_code}, "
                f"data={self.data}, "
                f"machine_confidence={self.machine_confidence})")
