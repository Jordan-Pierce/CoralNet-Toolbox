import warnings

import os
from collections import OrderedDict

from coralnet_toolbox.utilities import compose_volume_unit
from coralnet_toolbox.utilities import format_measurement
from coralnet_toolbox.utilities import convert_measurement
from coralnet_toolbox.utilities import compose_surface_area_unit

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def compute_builtin_fields(annotation, main_window):
    """Compute the read-only metadata derived from an annotation and its raster.

    These are the values the Confidence dock has always shown in its hover
    tooltip -- identity, confidence, geometry, 3D metrics and morphology. They
    are recomputed on demand rather than stored, so they can never go stale
    against the geometry they describe.

    Args:
        annotation: The annotation to describe.
        main_window: MainWindow, for the raster manager and the unit selector.

    Returns:
        tuple: (OrderedDict of display label -> formatted value string,
                set of scale units that could not be converted for display)
    """
    fields = OrderedDict()
    # Scale units encountered that cannot be converted to the display unit.
    # Collected so callers can explain why the unit dropdown appears to do
    # nothing, instead of silently showing unconverted numbers.
    unconvertible_units = set()

    if annotation is None:
        return fields, unconvertible_units

    target_unit = main_window.current_unit_scale

    # --- Identity ---
    fields['Annotation ID'] = str(annotation.id)
    fields['Annotation Type'] = type(annotation).__name__.replace('Annotation', '')

    if annotation.label:
        fields['Label'] = annotation.label.short_label_code
        if annotation.label.long_label_code != annotation.label.short_label_code:
            fields['Full Name'] = annotation.label.long_label_code

    # --- Confidence ---
    if annotation.user_confidence:
        top_label = max(annotation.user_confidence, key=annotation.user_confidence.get)
        top_confidence = annotation.user_confidence[top_label] * 100
        fields['User Confidence'] = f"{top_confidence:.1f}% ({top_label.short_label_code})"

    if annotation.machine_confidence:
        top_label = max(annotation.machine_confidence, key=annotation.machine_confidence.get)
        top_confidence = annotation.machine_confidence[top_label] * 100
        fields['Machine Confidence'] = f"{top_confidence:.1f}% ({top_label.short_label_code})"

    fields['Verified'] = 'Yes' if annotation.verified else 'No'

    # --- Source ---
    if annotation.image_path:
        fields['Source Image'] = os.path.basename(annotation.image_path)

    if annotation.cropped_image:
        width = annotation.cropped_image.width()
        height = annotation.cropped_image.height()
        fields['Cropped Dimensions'] = f"{width} x {height}"

    # --- Area ---
    try:
        scaled_area_data = annotation.get_scaled_area()
        if scaled_area_data:
            base_area_value, base_linear_unit = scaled_area_data
            # Convert, keeping the unit the value is genuinely in. If the
            # raster's scale units are not a convertible length (e.g. the
            # 'unknown' assigned to a world file with no CRS), the value is
            # left alone and labelled with its own unit instead of being
            # relabelled as the target.
            converted_area, area_unit, converted = convert_measurement(
                base_area_value, base_linear_unit, target_unit, squared=True
            )
            if not converted:
                unconvertible_units.add(base_linear_unit)
            fields['Area'] = f"{format_measurement(converted_area)} {area_unit}²"
        else:
            area = annotation.get_area()
            if area is not None:
                fields['Area'] = f"{format_measurement(area)} pixels²"
    except (NotImplementedError, AttributeError):
        pass  # No area method available

    # --- Perimeter ---
    try:
        scaled_perimeter_data = annotation.get_scaled_perimeter()
        if scaled_perimeter_data:
            base_perimeter_value, base_linear_unit = scaled_perimeter_data
            converted_perimeter, perim_unit, converted = convert_measurement(
                base_perimeter_value, base_linear_unit, target_unit
            )
            if not converted:
                unconvertible_units.add(base_linear_unit)
            fields['Perimeter'] = f"{format_measurement(converted_perimeter)} {perim_unit}"
        else:
            perimeter = annotation.get_perimeter()
            if perimeter is not None:
                fields['Perimeter'] = f"{format_measurement(perimeter)} pixels"
    except (NotImplementedError, AttributeError):
        pass  # No perimeter method available

    # --- 3D metrics, when the raster carries a z-channel ---
    raster = main_window.image_window.raster_manager.get_raster(annotation.image_path)

    if raster:
        # Lazily load the z_channel
        z_channel = raster.z_channel_lazy
        scale_x = raster.scale_x
        scale_y = raster.scale_y
        scale_units = raster.scale_units
        z_unit = raster.z_unit
        z_nodata = raster.z_nodata
        z_data_type = raster.z_data_type

        if z_channel is not None and scale_x is not None and scale_y is not None and scale_units is not None:
            try:
                volume = annotation.get_scaled_volume(z_channel, scale_x, scale_y, z_unit,
                                                      z_nodata=z_nodata, z_data_type=z_data_type)
                if volume is not None:
                    # The unit is composed from both scales rather than assumed:
                    # a relative z-channel (e.g. 'px') yields 'm² · px', not 'm³'
                    vol_units = compose_volume_unit(scale_units, z_unit)
                    fields['Volume'] = f"{format_measurement(volume)} {vol_units}"

                surface_area = annotation.get_scaled_surface_area(z_channel, scale_x, scale_y, z_unit,
                                                                  z_nodata=z_nodata,
                                                                  z_data_type=z_data_type)
                if surface_area is not None:
                    surf_units = compose_surface_area_unit(scale_units, z_unit)
                    fields['3D Surface Area'] = f"{format_measurement(surface_area)} {surf_units}"

                # 3D metrics are computed over valid pixels only, so a partly
                # empty z-channel under-reports. Surface that rather than hide it.
                coverage = annotation.get_z_coverage(z_channel,
                                                     z_nodata=z_nodata,
                                                     z_data_type=z_data_type)
                if coverage is not None and coverage < 1.0:
                    fields['Z Coverage'] = f"{coverage * 100:.0f}%"

            except Exception as e:
                print(f"Error calculating 3D metrics: {e}")

    # --- Morphology (only for annotation types that support it) ---
    try:
        morph_data = annotation.get_morphology()
        if morph_data:
            has_scale = 'units' in morph_data and morph_data['units'] is not None

            if has_scale and 'major_axis_scaled' in morph_data:
                base_unit = morph_data['units']
                major_scaled, axis_unit, converted = convert_measurement(
                    morph_data['major_axis_scaled'], base_unit, target_unit
                )
                minor_scaled, _, _ = convert_measurement(
                    morph_data['minor_axis_scaled'], base_unit, target_unit
                )
                if not converted:
                    unconvertible_units.add(base_unit)
                fields['Length'] = f"{format_measurement(major_scaled)} {axis_unit}"
                fields['Width'] = f"{format_measurement(minor_scaled)} {axis_unit}"
            else:
                if morph_data.get('major_axis') is not None:
                    fields['Length'] = f"{morph_data['major_axis']:.2f} px"
                if morph_data.get('minor_axis') is not None:
                    fields['Width'] = f"{morph_data['minor_axis']:.2f} px"

            if morph_data.get('orientation') is not None:
                fields['Orientation'] = f"{morph_data['orientation']:.1f}°"

            # Shape descriptors (unitless ratios)
            for key, display in (('aspect_ratio', 'Aspect Ratio'),
                                 ('roundness', 'Roundness'),
                                 ('circularity', 'Circularity'),
                                 ('compactness', 'Compactness'),
                                 ('solidity', 'Solidity'),
                                 ('convexity', 'Convexity'),
                                 ('elongation', 'Elongation'),
                                 ('rectangularity', 'Rectangularity'),
                                 ('eccentricity', 'Eccentricity')):
                if morph_data.get(key) is not None:
                    fields[display] = f"{morph_data[key]:.3f}"

            # Hull metrics
            if has_scale and 'hull_area_scaled' in morph_data:
                base_unit = morph_data['units']
                hull_area, hull_area_unit, converted = convert_measurement(
                    morph_data['hull_area_scaled'], base_unit, target_unit, squared=True
                )
                hull_perim, hull_perim_unit, _ = convert_measurement(
                    morph_data['hull_perimeter_scaled'], base_unit, target_unit
                )
                if not converted:
                    unconvertible_units.add(base_unit)
                fields['Hull Area'] = f"{format_measurement(hull_area)} {hull_area_unit}²"
                fields['Hull Perimeter'] = f"{format_measurement(hull_perim)} {hull_perim_unit}"
            else:
                if morph_data.get('hull_area') is not None:
                    fields['Hull Area'] = f"{morph_data['hull_area']:.2f} px²"
                if morph_data.get('hull_perimeter') is not None:
                    fields['Hull Perimeter'] = f"{morph_data['hull_perimeter']:.2f} px"

    except (NotImplementedError, AttributeError):
        pass  # No morphology method available

    # --- Scale basis ---
    # Makes it obvious which scale the numbers came from, and which unit they
    # are expressed in.
    if annotation.scale_x and annotation.scale_units:
        scale_text = f"{annotation.scale_x:.6g} {annotation.scale_units}/pixel"
        scale_source = getattr(raster, 'scale_source', None) if raster else None
        if scale_source:
            scale_text += f" ({scale_source})"
        fields['Scale'] = scale_text

    return fields, unconvertible_units


def format_unconvertible_note(unconvertible_units):
    """Build the explanatory note for scale units that could not be converted."""
    if not unconvertible_units:
        return ""
    units_text = ", ".join(sorted(str(unit) for unit in unconvertible_units))
    return (f"Scale units ({units_text}) are not a convertible length, "
            f"so values are shown unconverted and the unit selector has no effect.")
