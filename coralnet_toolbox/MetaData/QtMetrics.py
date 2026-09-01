import warnings

import math

import numpy as np

from coralnet_toolbox.utilities import convert_scale_units
from coralnet_toolbox.utilities import is_length_unit

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


METRICS_BY_ANNOTATION_TYPE = {
    'PatchAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': False,
        'bbox_min_y': False,
        'bbox_max_x': False,
        'bbox_max_y': False,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': False,
        'bbox_height': False,
        'equivalent_diameter': False,
        # Shape
        'major_axis': False,
        'minor_axis': False,
        'hull_area': False,
        'hull_perimeter': False,
        'aspect_ratio': False,
        'orientation': False,
        'roundness': False,
        'circularity': False,
        'compactness': False,
        'solidity': False,
        'convexity': False,
        'elongation': False,
        'rectangularity': False,
        'eccentricity': False,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'RectangleAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Rectangle now has full morphology support via get_morphology()
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'PolygonAnnotation': {
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Full morphology support
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
    'MultiPolygonAnnotation': {
        # Note: MultiPolygonAnnotation constituents (PolygonAnnotation) will be exported
        # individually with parent_annotation_id set. These metrics apply to each constituent.
        # Location
        'centroid_x': True,
        'centroid_y': True,
        'bbox_min_x': True,
        'bbox_min_y': True,
        'bbox_max_x': True,
        'bbox_max_y': True,
        # Size
        'area': True,
        'perimeter': True,
        'bbox_width': True,
        'bbox_height': True,
        'equivalent_diameter': True,
        # Shape - Full morphology support for constituent polygons
        'major_axis': True,
        'minor_axis': True,
        'hull_area': True,
        'hull_perimeter': True,
        'aspect_ratio': True,
        'orientation': True,
        'roundness': True,
        'circularity': True,
        'compactness': True,
        'solidity': True,
        'convexity': True,
        'elongation': True,
        'rectangularity': True,
        'eccentricity': True,
        # 3D Metrics
        'volume': True,
        'surface_area': True,
        'min_z': True,
        'max_z': True,
    },
}

# Organize metrics into categories for display
METRIC_CATEGORIES = {
    'Location': ['centroid_x', 'centroid_y', 'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y'],
    'Size': ['area', 'perimeter', 'bbox_width', 'bbox_height', 'equivalent_diameter'],
    'Shape': ['major_axis', 'minor_axis', 'hull_area', 'hull_perimeter', 'aspect_ratio',
              'orientation', 'roundness', 'circularity', 'compactness', 'solidity',
              'convexity', 'elongation', 'rectangularity', 'eccentricity'],
    '3D Metrics': ['volume', 'surface_area', 'min_z', 'max_z'],
}

# All metrics in order
ALL_METRICS = (
    METRIC_CATEGORIES['Location'] +
    METRIC_CATEGORIES['Size'] +
    METRIC_CATEGORIES['Shape'] +
    METRIC_CATEGORIES['3D Metrics']
)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def calculate_metrics_for_annotation(annotation, selected_metrics, z_channel=None, z_unit=None,
                                     z_nodata=None, z_data_type=None):
    """
    Calculate the derived metrics for a single annotation.

    Args:
        annotation: The annotation object
        selected_metrics: List of metric names to calculate
        z_channel: The z-channel data for 3D metrics (optional)
        z_unit: The unit of the z-channel data (optional)
        z_nodata: Sentinel marking missing z measurements (optional)
        z_data_type: 'depth' or 'elevation' (optional)

    Returns:
        dict: Dictionary with metric values (pixel and meters columns)
    """
    annotation_type = type(annotation).__name__
    type_metrics = METRICS_BY_ANNOTATION_TYPE.get(annotation_type, {})

    result = {}

    # Get scale info if available
    scale_x = annotation.scale_x
    scale_y = annotation.scale_y
    scale_units = annotation.scale_units
    # The '(meters)' columns are only meaningful when the annotation's scale
    # unit can actually be converted to metres. For anything else (e.g. the
    # 'degree' of a lon/lat world file) convert_scale_units is a no-op, which
    # would fill a metres column with unconverted values.
    has_scale = (scale_x is not None and scale_y is not None
                 and is_length_unit(scale_units))

    # Calculate conversion factor to meters if scale is available
    to_meters_factor = 1.0
    if has_scale and scale_units:
        try:
            to_meters_factor = convert_scale_units(1.0, scale_units, 'metre')
        except Exception:
            to_meters_factor = 1.0

    # Get basic geometry info
    center_xy = annotation.center_xy
    top_left = annotation.get_bounding_box_top_left()
    bottom_right = annotation.get_bounding_box_bottom_right()

    # Calculate bbox dimensions
    bbox_width = abs(bottom_right.x() - top_left.x()) if top_left and bottom_right else None
    bbox_height = abs(bottom_right.y() - top_left.y()) if top_left and bottom_right else None

    # Get morphology data - always try to get it now that RectangleAnnotation supports it
    morph_data = annotation.get_morphology()

    # Process each selected metric
    for metric in selected_metrics:
        if metric not in type_metrics:
            continue

        is_applicable = type_metrics.get(metric, False)
        pixel_value = None
        meter_value = None

        if is_applicable:
            try:
                # Location metrics
                if metric == 'centroid_x':
                    pixel_value = float(center_xy.x()) if center_xy else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'centroid_y':
                    pixel_value = float(center_xy.y()) if center_xy else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_y * to_meters_factor
                elif metric == 'bbox_min_x':
                    pixel_value = float(top_left.x()) if top_left else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'bbox_min_y':
                    pixel_value = float(top_left.y()) if top_left else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_y * to_meters_factor
                elif metric == 'bbox_max_x':
                    pixel_value = float(bottom_right.x()) if bottom_right else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'bbox_max_y':
                    pixel_value = float(bottom_right.y()) if bottom_right else None
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_y * to_meters_factor

                # Size metrics
                elif metric == 'area':
                    pixel_value = annotation.get_area()
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * scale_y * (to_meters_factor ** 2)
                elif metric == 'perimeter':
                    pixel_value = annotation.get_perimeter()
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'bbox_width':
                    pixel_value = bbox_width
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'bbox_height':
                    pixel_value = bbox_height
                    if pixel_value is not None and has_scale:
                        meter_value = pixel_value * scale_y * to_meters_factor
                elif metric == 'equivalent_diameter':
                    area = annotation.get_area()
                    if area is not None and area > 0:
                        pixel_value = math.sqrt(4 * area / math.pi)
                        if has_scale:
                            area_meters = area * scale_x * scale_y * (to_meters_factor ** 2)
                            meter_value = math.sqrt(4 * area_meters / math.pi)

                # Shape metrics from morphology - all pulled from get_morphology() now
                elif metric in ['major_axis', 'minor_axis', 'hull_perimeter']:
                    # Linear measurements
                    if morph_data:
                        pixel_value = morph_data.get(metric)
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * to_meters_factor
                elif metric == 'hull_area':
                    # Area measurement
                    if morph_data:
                        pixel_value = morph_data.get(metric)
                        if pixel_value is not None and has_scale:
                            meter_value = pixel_value * scale_x * scale_y * (to_meters_factor ** 2)
                elif metric in ['aspect_ratio', 'orientation', 'roundness', 'circularity', 
                                'compactness', 'solidity', 'convexity', 'elongation', 
                                'rectangularity', 'eccentricity']:
                    # Unitless ratios - pull directly from morphology
                    if morph_data:
                        pixel_value = morph_data.get(metric)
                    meter_value = pixel_value  # Unitless

                # 3D Metrics
                elif metric == 'volume':
                    if z_channel is not None and has_scale:
                        # Convert scales to meters/pixel as required by get_scaled_volume
                        scale_x_meters = scale_x * to_meters_factor
                        scale_y_meters = scale_y * to_meters_factor
                        volume = annotation.get_scaled_volume(z_channel, scale_x_meters, scale_y_meters,
                                                              z_unit, z_nodata=z_nodata,
                                                              z_data_type=z_data_type)
                        if volume is not None:
                            # Volume is already in the correct units (cubic meters)
                            meter_value = volume
                            # Calculate pixel-based volume: sum of z-values in pixels
                            try:
                                z_slice, poly_mask, valid_mask = annotation._get_valid_z_slice_and_mask(
                                    z_channel, z_nodata=z_nodata, z_data_type=z_data_type
                                )
                                mask = poly_mask & valid_mask
                                if z_slice.size > 0 and mask.size > 0 and np.any(mask):
                                    pixel_value = float(np.sum(z_slice[mask]))
                            except Exception:
                                pixel_value = None
                elif metric == 'surface_area':
                    if z_channel is not None and has_scale:
                        # Convert scales to meters/pixel as required by get_scaled_surface_area
                        scale_x_meters = scale_x * to_meters_factor
                        scale_y_meters = scale_y * to_meters_factor
                        surf_area = annotation.get_scaled_surface_area(z_channel, 
                                                                       scale_x_meters, 
                                                                       scale_y_meters,
                                                                       z_unit,
                                                                       z_nodata=z_nodata,
                                                                       z_data_type=z_data_type)
                        if surf_area is not None:
                            # Surface area is already in the correct units (square meters)
                            meter_value = surf_area
                            # Calculate pixel-based surface area: sum of 3D surface elements
                            try:
                                z_slice, poly_mask, valid_mask = annotation._get_valid_z_slice_and_mask(
                                    z_channel, z_nodata=z_nodata, z_data_type=z_data_type
                                )
                                mask = poly_mask & valid_mask
                                if z_slice.size > 0 and mask.size > 0 and np.any(mask):
                                    # Fill holes so nodata does not corrupt neighbouring slopes
                                    if not np.all(valid_mask):
                                        z_slice = z_slice.astype(np.float32, copy=True)
                                        z_slice[~valid_mask] = float(np.mean(z_slice[valid_mask]))
                                    # Calculate gradients in pixel space
                                    dz_dy, dz_dx = np.gradient(z_slice)
                                    # Surface area multiplier for each pixel
                                    multiplier = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
                                    # Sum surface elements inside mask (each pixel has area = 1 in pixel space)
                                    pixel_value = float(np.sum(multiplier[mask]))
                            except Exception:
                                pixel_value = None
                elif metric == 'min_z':
                    if z_channel is not None:
                        z_data = annotation.get_min_z(z_channel, scale_x, z_unit,
                                                      z_nodata=z_nodata, z_data_type=z_data_type)
                        if z_data:
                            pixel_value = z_data.get('pixels')
                            meter_value = z_data.get('meters')
                elif metric == 'max_z':
                    if z_channel is not None:
                        z_data = annotation.get_max_z(z_channel, scale_x, z_unit,
                                                      z_nodata=z_nodata, z_data_type=z_data_type)
                        if z_data:
                            pixel_value = z_data.get('pixels')
                            meter_value = z_data.get('meters')

            except Exception as e:
                # Log the error but continue with None values
                print(f"Error calculating {metric} for annotation {annotation.id}: {e}")

        # Round values
        if pixel_value is not None and isinstance(pixel_value, float):
            pixel_value = round(pixel_value, 4)
        if meter_value is not None and isinstance(meter_value, float):
            meter_value = round(meter_value, 4)

        # Add to result
        result[f"{metric} (pixels)"] = pixel_value
        result[f"{metric} (meters)"] = meter_value

    return result
