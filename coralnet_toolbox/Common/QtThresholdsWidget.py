"""
QtThresholdsWidget - Reusable widget for threshold controls
"""

import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QGroupBox, QFormLayout, QLabel, QSizePolicy,
                             QSlider, QSpinBox)

from coralnet_toolbox.utilities import convert_scale_units
from coralnet_toolbox.utilities import is_length_unit


# The area threshold is expressed either as a share of the image or as a real
# world area. A share applies to any raster but means a different physical size
# on each one; a real-world area is the same size everywhere, but needs the
# raster to carry a scale.
AREA_MODE_FRACTION = 'fraction'
AREA_MODE_METRIC = 'metric'

AREA_SLIDER_STEPS = 1000

# log10 bounds of each mode's slider travel. Both are logarithmic because a
# linear slider cannot span the sizes involved: one tick of the old linear
# 0-100 slider was 1% of the image area, a 2400px square on a 24k ortho, so
# every real object sat below the first tick.
AREA_SLIDER_RANGES = {
    AREA_MODE_FRACTION: (-6.0, 0.0),   # 1e-6 up to all of the image area
    AREA_MODE_METRIC: (-4.0, 4.0),     # 1 cm2 up to 1 hectare, held in m2
}

# Seed values for a mode when nothing better can be derived on a switch.
AREA_MODE_DEFAULTS = {
    AREA_MODE_FRACTION: (0.0, 0.40),
    AREA_MODE_METRIC: (0.0, 100.0),    # m2
}


def area_slider_tick(mode: str = AREA_MODE_FRACTION) -> int:
    """Slider tick spacing for `mode`: one tick per decade."""
    lo, hi = AREA_SLIDER_RANGES.get(mode, AREA_SLIDER_RANGES[AREA_MODE_FRACTION])
    return max(1, int(AREA_SLIDER_STEPS / (hi - lo)))


def area_slider_to_value(value: int, mode: str = AREA_MODE_FRACTION) -> float:
    """Map a slider position to a threshold value in `mode`'s unit."""
    lo, hi = AREA_SLIDER_RANGES.get(mode, AREA_SLIDER_RANGES[AREA_MODE_FRACTION])
    if value <= 0:
        return 0.0
    if value >= AREA_SLIDER_STEPS:
        return 10.0 ** hi
    return 10.0 ** (lo + (hi - lo) * value / AREA_SLIDER_STEPS)


def area_value_to_slider(value: float, mode: str = AREA_MODE_FRACTION) -> int:
    """Map a threshold value in `mode`'s unit back to a slider position."""
    lo, hi = AREA_SLIDER_RANGES.get(mode, AREA_SLIDER_RANGES[AREA_MODE_FRACTION])
    if not value or value <= 0:
        return 0
    exponent = math.log10(value)
    if exponent <= lo:
        return 0
    if exponent >= hi:
        return AREA_SLIDER_STEPS
    return int(round(AREA_SLIDER_STEPS * (exponent - lo) / (hi - lo)))


def format_area_fraction(fraction: float) -> str:
    """Render an area fraction as a percentage legible across the whole log range."""
    if not fraction or fraction <= 0.0:
        return "0%"
    if fraction >= 1.0:
        return "100%"
    percent = fraction * 100.0
    if percent >= 1.0:
        return f"{percent:.1f}%"
    if percent >= 0.001:
        return f"{percent:.3f}%"
    return f"{percent:.1e}%"


def format_area_metric_range(min_m2: float, max_m2: float) -> str:
    """Render a real-world area range, picking one unit for both ends.

    The unit is chosen from the larger bound so the two numbers stay
    comparable; mixing them ("0 cm2 - 2,314 m2") reads as a mistake.
    """
    largest = max(min_m2, max_m2)
    if largest < 1.0:
        return f"{min_m2 * 1e4:,.1f} - {max_m2 * 1e4:,.1f} cm\u00b2"
    if largest < 1e6:
        return f"{min_m2:,.2f} - {max_m2:,.2f} m\u00b2"
    return f"{min_m2 / 1e6:,.3f} - {max_m2 / 1e6:,.3f} km\u00b2"


def raster_metrics(raster):
    """Pixel area and square metres per pixel for one raster.

    Returns (image_area_px, m2_per_px); either element is None when it cannot
    be established. Most rasters carry no scale at all, and a world file with
    no CRS can leave scale_units as 'degree' - a real unit, but not a length,
    so it must not be multiplied through as if it were metres.
    """
    if raster is None:
        return None, None

    try:
        if not raster.width or not raster.height:
            return None, None
        image_area = float(raster.width) * float(raster.height)
    except Exception:
        return None, None

    m2_per_px = None
    try:
        units = raster.scale_units
        if raster.scale_x and raster.scale_y and units and is_length_unit(units):
            to_metres = convert_scale_units(1.0, units, 'metre')
            m2_per_px = (float(raster.scale_x) * to_metres) * (float(raster.scale_y) * to_metres)
    except Exception:
        m2_per_px = None

    return image_area, m2_per_px


def current_raster_metrics(main_window):
    """raster_metrics for whichever raster is on display."""
    try:
        image_path = main_window.annotation_window.current_image_path
        if not image_path:
            return None, None
        raster = main_window.image_window.raster_manager.get_raster(image_path)
    except Exception:
        return None, None

    return raster_metrics(raster)


def resolve_area_bounds_px(min_value, max_value, mode, image_area, m2_per_px):
    """The area threshold as absolute px2 bounds for one raster.

    Returns (min_px, max_px), or None when the threshold cannot be evaluated
    against this raster - a real-world bound on a raster carrying no scale.
    Callers must then accept everything: we cannot judge the criterion, and
    silently dropping every detection would be far worse than not filtering.
    """
    if mode == AREA_MODE_METRIC:
        if not m2_per_px:
            return None
        return min_value / m2_per_px, max_value / m2_per_px

    if not image_area:
        return None
    return min_value * image_area, max_value * image_area


def convert_area_bounds(min_value, max_value, from_mode, to_mode, image_area, m2_per_px):
    """Carry a threshold across a mode switch, preserving the physical size.

    Falls back to the target mode's defaults when the open raster cannot bridge
    the two - no scale, or no image at all - since a known-sane starting point
    beats an arbitrary number the user did not choose.
    """
    if from_mode == to_mode:
        return min_value, max_value

    if image_area and m2_per_px:
        image_m2 = image_area * m2_per_px
        if image_m2:
            if to_mode == AREA_MODE_METRIC:
                return min_value * image_m2, max_value * image_m2
            return min_value / image_m2, max_value / image_m2

    return AREA_MODE_DEFAULTS[to_mode]


def get_area_mode(main_window) -> str:
    """The active area threshold mode, defaulting to the image-share mode.

    Tolerates a main window that predates the mode so a caller never has to
    guard the attribute itself.
    """
    try:
        mode = main_window.get_area_thresh_mode()
    except Exception:
        return AREA_MODE_FRACTION
    return mode if mode in AREA_SLIDER_RANGES else AREA_MODE_FRACTION


def format_area_range(min_value: float, max_value: float,
                      mode: str = AREA_MODE_FRACTION) -> str:
    """The bounds on their own, in the active unit."""
    if mode == AREA_MODE_METRIC:
        return format_area_metric_range(min_value, max_value)
    return f"{format_area_fraction(min_value)} - {format_area_fraction(max_value)}"


def format_area_equivalent(min_value: float, max_value: float,
                           image_area: float = None, m2_per_px: float = None,
                           mode: str = AREA_MODE_FRACTION) -> str:
    """The same bounds restated for the open image.

    Returns "" when the open raster cannot say anything useful. A percentage is
    unreadable at the bottom of a log scale - 0.001% of an image says nothing
    about whether it will catch a coral colony - so this is what makes the
    setting concrete.
    """
    if mode == AREA_MODE_METRIC:
        if not image_area:
            return ""
        if not m2_per_px:
            return "no scale on this image, so the area filter is inactive"
        return (f"~{min_value / m2_per_px:,.0f} - "
                f"{max_value / m2_per_px:,.0f} px\u00b2")

    if not image_area:
        return ""

    min_px = min_value * image_area
    max_px = max_value * image_area
    if m2_per_px:
        return "~" + format_area_metric_range(min_px * m2_per_px, max_px * m2_per_px)
    return f"~{min_px:,.0f} - {max_px:,.0f} px\u00b2"


def format_area_label(min_value: float, max_value: float,
                      image_area: float = None, m2_per_px: float = None,
                      mode: str = AREA_MODE_FRACTION) -> str:
    """Compact form for the threshold panel.

    The equivalents live in the status bar instead - spelled out in the panel
    they set the width of the whole dialog. Only the inactive warning is kept,
    abbreviated, because a filter that is silently doing nothing has to say so
    where the controls are.
    """
    text = format_area_range(min_value, max_value, mode)
    if mode == AREA_MODE_METRIC and image_area and not m2_per_px:
        text += "  (inactive)"
    return text


def format_area_status(min_value: float, max_value: float,
                       image_area: float = None, m2_per_px: float = None,
                       mode: str = AREA_MODE_FRACTION) -> str:
    """The full reading, for the status bar."""
    text = f"Area threshold: {format_area_range(min_value, max_value, mode)}"
    equivalent = format_area_equivalent(min_value, max_value, image_area, m2_per_px, mode)
    if equivalent:
        text += f"  ({equivalent})"
    return text


def set_area_mode_availability(combo, m2_per_px) -> None:
    """Grey out the real-world entry when the open raster carries no scale.

    A combo already sitting on real-world keeps it: silently switching on
    navigation would convert the user's bounds behind their back. The entry
    stays selected but disabled, and the label reports the filter as inactive.
    """
    if combo is None:
        return

    index = combo.findData(AREA_MODE_METRIC)
    if index < 0:
        return

    model = combo.model()
    item = model.item(index) if hasattr(model, 'item') else None
    if item is None:
        return

    item.setEnabled(bool(m2_per_px))
    item.setToolTip("" if m2_per_px else
                    "This image has no scale, so a real-world area cannot be measured. "
                    "Set one with the Scale tool, or open a georeferenced raster.")


# How long the area reading lingers in the status bar after the last change.
AREA_STATUS_TIMEOUT_MS = 5000


class ThresholdsWidget(QGroupBox):
    """
    A reusable widget that provides threshold controls (max detections, boundary detections,
    uncertainty, IoU, area min/max).
    This widget can be configured to show only the controls needed for a specific use case.
    
    :param main_window: MainWindow object to sync threshold values
    :param show_max_detections: Whether to show the max detections spinbox
    :param show_boundary: Whether to show the boundary detections combo box
    :param show_uncertainty: Whether to show the uncertainty threshold slider
    :param show_iou: Whether to show the IoU threshold slider
    :param show_area: Whether to show the area threshold sliders (min and max)
    :param title: Title for the group box (default: "Thresholds")
    :param parent: Parent widget
    """
    
    def __init__(self, main_window, show_max_detections=False,
                 show_uncertainty=True, show_iou=False, show_area=False,
                 title="Thresholds", parent=None, show_boundary=False):
        super().__init__(title, parent)
        
        self.main_window = main_window
        
        # Initialize threshold values from main window
        self.max_detections = main_window.get_max_detections()
        self.boundary_tolerance = main_window.get_boundary_tolerance()
        self.uncertainty_thresh = main_window.get_uncertainty_thresh()
        self.iou_thresh = main_window.get_iou_thresh()
        min_val, max_val = main_window.get_area_thresh()
        self.area_thresh_min = min_val
        self.area_thresh_max = max_val
        self.area_thresh_mode = get_area_mode(main_window)
        
        # Create the layout
        layout = QFormLayout()

        def stretch_field(field_widget):
            """Let a field grow with the form instead of sitting at its sizeHint.

            Only needed for the combo boxes: QFormLayout's default growth policy
            here grows Expanding fields only, which sliders already are.
            """
            field_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def apply_row_tooltip(field_widget, tooltip_text):
            field_widget.setToolTip(tooltip_text)
            field_widget.setStatusTip(tooltip_text)
            label_widget = layout.labelForField(field_widget)
            if label_widget is not None:
                label_widget.setToolTip(tooltip_text)
                label_widget.setStatusTip(tooltip_text)
        
        # Max detections spinbox
        if show_max_detections:
            self.max_detections_spinbox = QSpinBox()
            self.max_detections_spinbox.setRange(1, 10000)
            self.max_detections_spinbox.setValue(main_window.get_max_detections())
            stretch_field(self.max_detections_spinbox)
            self.max_detections_spinbox.valueChanged.connect(self._update_max_detections)
            main_window.maxDetectionsChanged.connect(self._on_max_detections_changed)
            layout.addRow("Max Detections:", self.max_detections_spinbox)
            apply_row_tooltip(
                self.max_detections_spinbox,
                "Maximum number of detections kept after Ultralytics non-max suppression. Lower "
                "values reduce clutter and processing time; higher values allow more candidates to "
                "survive into annotation creation.")

        if hasattr(main_window, 'boundaryToleranceChanged'):
            main_window.boundaryToleranceChanged.connect(self._on_boundary_tolerance_changed)
        if show_boundary:
            # Boundary detections controls
            self.boundary_tolerance_combo = QComboBox()
            self.boundary_tolerance_combo.addItems([
                "Keep",
                "Ignore",
            ])
            self.boundary_tolerance_combo.setCurrentIndex(0 if self.boundary_tolerance else 1)
            stretch_field(self.boundary_tolerance_combo)
            self.boundary_tolerance_combo.currentIndexChanged.connect(self._update_boundary_tolerance)
            layout.addRow("Boundary Detections", self.boundary_tolerance_combo)
            apply_row_tooltip(
                self.boundary_tolerance_combo,
                "Choose whether detections that touch a work-area edge should be preserved. Keep retains "
                "cut-off objects, while Ignore removes them to reduce seam duplicates across tiles.")
        
        # Uncertainty threshold controls
        if show_uncertainty:
            self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
            self.uncertainty_threshold_slider.setRange(0, 100)
            self.uncertainty_threshold_slider.setValue(int(self.uncertainty_thresh * 100))
            self.uncertainty_threshold_slider.setTickPosition(QSlider.TicksBelow)
            self.uncertainty_threshold_slider.setTickInterval(10)
            self.uncertainty_threshold_slider.valueChanged.connect(self._update_uncertainty_label)
            main_window.uncertaintyChanged.connect(self._on_uncertainty_changed)
            self.uncertainty_threshold_label = QLabel(f"{self.uncertainty_thresh:.2f}")
            layout.addRow("Uncertainty Threshold", self.uncertainty_threshold_slider)
            layout.addRow("", self.uncertainty_threshold_label)
            apply_row_tooltip(
                self.uncertainty_threshold_slider,
                "Minimum confidence required before a prediction is accepted as a normal annotation. "
                "Predictions below this value are treated as Review and can be surfaced for manual "
                "inspection.")
            self.uncertainty_threshold_label.setToolTip(
                "Current uncertainty threshold value, shown as a 0.00 to 1.00 confidence cutoff.")
        
        # IoU threshold controls
        if show_iou:
            self.iou_threshold_slider = QSlider(Qt.Horizontal)
            self.iou_threshold_slider.setRange(0, 100)
            self.iou_threshold_slider.setValue(int(self.iou_thresh * 100))
            self.iou_threshold_slider.setTickPosition(QSlider.TicksBelow)
            self.iou_threshold_slider.setTickInterval(10)
            self.iou_threshold_slider.valueChanged.connect(self._update_iou_label)
            main_window.iouChanged.connect(self._on_iou_changed)
            self.iou_threshold_label = QLabel(f"{self.iou_thresh:.2f}")
            layout.addRow("IoU Threshold", self.iou_threshold_slider)
            layout.addRow("", self.iou_threshold_label)
            apply_row_tooltip(
                self.iou_threshold_slider,
                "Intersection-over-Union threshold used by non-max suppression. Higher values keep "
                "more overlapping detections; lower values remove duplicates more aggressively.")
            self.iou_threshold_label.setToolTip(
                "Current IoU threshold value used for non-max suppression.")
        
        # Area threshold controls
        if show_area:
            self.area_mode_combo = QComboBox()
            self.area_mode_combo.addItem("Image %", AREA_MODE_FRACTION)
            self.area_mode_combo.addItem("Real-world", AREA_MODE_METRIC)
            self.area_mode_combo.setCurrentIndex(
                max(0, self.area_mode_combo.findData(self.area_thresh_mode)))
            stretch_field(self.area_mode_combo)
            self.area_mode_combo.currentIndexChanged.connect(self._update_area_mode)
            layout.addRow("Area Units", self.area_mode_combo)
            apply_row_tooltip(
                self.area_mode_combo,
                "Image %: the bounds are a share of each image's area, so the same setting picks out "
                "a different physical size on every raster.\n\n"
                "Real-world: the bounds are an absolute area, so one setting means the same size "
                "across a whole dataset. Requires the raster to carry a scale - on an unscaled image "
                "the area filter is skipped rather than applied wrongly.")

            area_tick = area_slider_tick(self.area_thresh_mode)
            self.area_threshold_min_slider = QSlider(Qt.Horizontal)
            self.area_threshold_min_slider.setRange(0, AREA_SLIDER_STEPS)
            self.area_threshold_min_slider.setValue(
                area_value_to_slider(self.area_thresh_min, self.area_thresh_mode))
            self.area_threshold_min_slider.setTickPosition(QSlider.TicksBelow)
            # One tick per decade.
            self.area_threshold_min_slider.setTickInterval(area_tick)
            self.area_threshold_min_slider.valueChanged.connect(self._update_area_label)

            self.area_threshold_max_slider = QSlider(Qt.Horizontal)
            self.area_threshold_max_slider.setRange(0, AREA_SLIDER_STEPS)
            self.area_threshold_max_slider.setValue(
                area_value_to_slider(self.area_thresh_max, self.area_thresh_mode))
            self.area_threshold_max_slider.setTickPosition(QSlider.TicksBelow)
            self.area_threshold_max_slider.setTickInterval(area_tick)
            self.area_threshold_max_slider.valueChanged.connect(self._update_area_label)

            main_window.areaChanged.connect(self._on_area_changed)
            if hasattr(main_window, 'areaModeChanged'):
                main_window.areaModeChanged.connect(self._on_area_mode_changed)

            self.area_threshold_label = QLabel(self._area_label_text())
            layout.addRow("Area Threshold Min", self.area_threshold_min_slider)
            layout.addRow("Area Threshold Max", self.area_threshold_max_slider)
            layout.addRow("", self.area_threshold_label)
            apply_row_tooltip(
                self.area_threshold_min_slider,
                "Lower bound of the annotation area filter, as a fraction of the whole image area. "
                "Objects smaller than this are removed after confidence and IoU filtering.\n\n"
                "The slider is logarithmic: each tick is one decade, spanning 0.0001% up to 100% of "
                "the image. A linear slider could not reach the sizes real objects occupy in a large "
                "orthomosaic.")
            apply_row_tooltip(
                self.area_threshold_max_slider,
                "Upper bound of the annotation area filter, as a fraction of the whole image area. "
                "Objects larger than this are removed after confidence and IoU filtering.\n\n"
                "The slider is logarithmic: each tick is one decade, spanning 0.0001% up to 100% of "
                "the image.")
            self.area_threshold_label.setToolTip(
                "Current area filter range, as a percentage of the image area. When an image is open "
                "the equivalent bounds are also shown - in real-world units if that raster is scaled, "
                "otherwise in pixels.")
            self._refresh_area_mode_availability()
            try:
                main_window.image_window.imageLoaded.connect(self._on_image_loaded)
            except Exception:
                pass
        
        self.setLayout(layout)
        
    def _update_max_detections(self, value):
        """Update max detections value"""
        self.max_detections = value
        self.main_window.update_max_detections(value)

    def _update_boundary_tolerance(self, value):
        """Update boundary detection handling value"""
        if isinstance(value, bool):
            keep_boundary_detections = value
        else:
            keep_boundary_detections = int(value) == 0
        self.boundary_tolerance = keep_boundary_detections
        self.main_window.update_boundary_tolerance(keep_boundary_detections)
    
    def _update_uncertainty_label(self, value):
        """Update uncertainty threshold and label"""
        value = value / 100.0
        self.uncertainty_thresh = value
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_threshold_label.setText(f"{value:.2f}")
    
    def _update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        self.main_window.update_iou_thresh(value)
        self.iou_threshold_label.setText(f"{value:.2f}")
    
    def _area_label_text(self):
        """The compact reading shown beside the sliders."""
        image_area, m2_per_px = current_raster_metrics(self.main_window)
        return format_area_label(self.area_thresh_min, self.area_thresh_max,
                                 image_area, m2_per_px, self.area_thresh_mode)

    def _push_area_status(self):
        """Put the full reading in the status bar.

        Only fired by a deliberate change - a slider move or a unit switch - so
        navigating between images does not spam the status bar.
        """
        try:
            image_area, m2_per_px = current_raster_metrics(self.main_window)
            self.main_window.status_bar.showMessage(
                format_area_status(self.area_thresh_min, self.area_thresh_max,
                                   image_area, m2_per_px, self.area_thresh_mode),
                AREA_STATUS_TIMEOUT_MS)
        except Exception:
            pass

    def _refresh_area_mode_availability(self):
        """Match the units combo to whether the open raster has a scale."""
        if hasattr(self, 'area_mode_combo'):
            _, m2_per_px = current_raster_metrics(self.main_window)
            set_area_mode_availability(self.area_mode_combo, m2_per_px)

    def _sync_area_widgets(self):
        """Re-seed the sliders and label from the active mode and values."""
        if not hasattr(self, 'area_threshold_min_slider'):
            return
        tick = area_slider_tick(self.area_thresh_mode)
        for slider, value in ((self.area_threshold_min_slider, self.area_thresh_min),
                              (self.area_threshold_max_slider, self.area_thresh_max)):
            slider.blockSignals(True)
            slider.setTickInterval(tick)
            slider.setValue(area_value_to_slider(value, self.area_thresh_mode))
            slider.blockSignals(False)
        self.area_threshold_label.setText(self._area_label_text())

    def _update_area_mode(self, index):
        """Hand a mode change to the main window, which owns the conversion."""
        mode = self.area_mode_combo.itemData(index)
        if mode:
            self.main_window.update_area_thresh_mode(mode)

    def _on_area_mode_changed(self, mode):
        """Adopt a mode change made elsewhere."""
        self.area_thresh_mode = mode
        if hasattr(self, 'area_mode_combo'):
            self.area_mode_combo.blockSignals(True)
            self.area_mode_combo.setCurrentIndex(max(0, self.area_mode_combo.findData(mode)))
            self.area_mode_combo.blockSignals(False)
        self.area_thresh_min, self.area_thresh_max = self.main_window.get_area_thresh()
        self._sync_area_widgets()
        self._push_area_status()

    def _on_image_loaded(self, *args):
        """Redraw the area label when the displayed raster changes.

        The hint is per-raster - pixel bounds scale with the image, and only
        some rasters carry a scale at all - so a dialog left open across a
        navigation would otherwise keep showing the previous image's numbers.
        """
        if hasattr(self, 'area_threshold_label'):
            self.area_threshold_label.setText(self._area_label_text())
        self._refresh_area_mode_availability()

    def _update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val = self.area_threshold_min_slider.value()
        max_val = self.area_threshold_max_slider.value()
        if min_val > max_val:
            min_val = max_val
            self.area_threshold_min_slider.setValue(min_val)
        self.area_thresh_min = area_slider_to_value(min_val, self.area_thresh_mode)
        self.area_thresh_max = area_slider_to_value(max_val, self.area_thresh_mode)
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_label.setText(self._area_label_text())
        self._push_area_status()
    
    def initialize_thresholds(self):
        """
        Initialize threshold sliders with current values from main window.
        This should be called in the parent dialog's showEvent.
        """
        if hasattr(self, 'max_detections_spinbox'):
            current_value = self.main_window.get_max_detections()
            self.max_detections_spinbox.setValue(current_value)
            self.max_detections = current_value

        current_value = self.main_window.get_boundary_tolerance()
        self.boundary_tolerance = current_value
        if hasattr(self, 'boundary_tolerance_combo'):
            self.boundary_tolerance_combo.blockSignals(True)
            self.boundary_tolerance_combo.setCurrentIndex(0 if current_value else 1)
            self.boundary_tolerance_combo.blockSignals(False)
        
        if hasattr(self, 'uncertainty_threshold_slider'):
            current_value = self.main_window.get_uncertainty_thresh()
            self.uncertainty_threshold_slider.setValue(int(current_value * 100))
            self.uncertainty_thresh = current_value
        
        if hasattr(self, 'iou_threshold_slider'):
            current_value = self.main_window.get_iou_thresh()
            self.iou_threshold_slider.setValue(int(current_value * 100))
            self.iou_thresh = current_value
        
        if hasattr(self, 'area_threshold_min_slider') and hasattr(self, 'area_threshold_max_slider'):
            self.area_thresh_min, self.area_thresh_max = self.main_window.get_area_thresh()
            self.area_thresh_mode = get_area_mode(self.main_window)
            if hasattr(self, 'area_mode_combo'):
                self.area_mode_combo.blockSignals(True)
                self.area_mode_combo.setCurrentIndex(
                    max(0, self.area_mode_combo.findData(self.area_thresh_mode)))
                self.area_mode_combo.blockSignals(False)
            # Refresh here, not only on slider moves: the reading is relative to
            # whichever image is open when the dialog is shown.
            self._refresh_area_mode_availability()
            self._sync_area_widgets()
    
    def get_max_detections(self):
        """Get the current max detections value"""
        return self.max_detections

    def get_boundary_tolerance(self):
        """Get whether detections on boundaries should be kept"""
        return self.boundary_tolerance
    
    def get_uncertainty_thresh(self):
        """Get the current uncertainty threshold value"""
        return self.uncertainty_thresh
    
    def get_iou_thresh(self):
        """Get the current IoU threshold value"""
        return self.iou_thresh
    
    def get_area_thresh_min(self):
        """Get the current minimum area threshold value"""
        return self.area_thresh_min
    
    def get_area_thresh_max(self):
        """Get the current maximum area threshold value"""
        return self.area_thresh_max

    def get_area_thresh_mode(self):
        """Get the unit the area threshold is expressed in"""
        return self.area_thresh_mode

    def set_area_enabled(self, enabled):
        """Enable or disable the area controls as a group, row labels included.

        For dialogs where the area filter only applies under some other option:
        greying the controls says so, where leaving them live but ignored would
        not.
        """
        if not hasattr(self, 'area_threshold_min_slider'):
            return

        layout = self.layout()
        widgets = [self.area_threshold_min_slider,
                   self.area_threshold_max_slider,
                   self.area_threshold_label]
        if hasattr(self, 'area_mode_combo'):
            widgets.append(self.area_mode_combo)

        for widget in widgets:
            widget.setEnabled(enabled)
            label = layout.labelForField(widget) if layout is not None else None
            if label is not None:
                label.setEnabled(enabled)
    
    def _on_max_detections_changed(self, value):
        """Update spinbox when MainWindow changes"""
        if hasattr(self, 'max_detections_spinbox'):
            self.max_detections_spinbox.blockSignals(True)  # Prevent recursive signals
            self.max_detections_spinbox.setValue(value)
            self.max_detections = value
            self.max_detections_spinbox.blockSignals(False)

    def _on_boundary_tolerance_changed(self, value):
        """Update combo box when MainWindow changes"""
        self.boundary_tolerance = value
        if hasattr(self, 'boundary_tolerance_combo'):
            self.boundary_tolerance_combo.blockSignals(True)
            self.boundary_tolerance_combo.setCurrentIndex(0 if value else 1)
            self.boundary_tolerance_combo.blockSignals(False)
    
    def _on_uncertainty_changed(self, value):
        """Update slider/label when MainWindow changes"""
        if hasattr(self, 'uncertainty_threshold_slider'):
            self.uncertainty_threshold_slider.blockSignals(True)
            self.uncertainty_threshold_slider.setValue(int(value * 100))
            self.uncertainty_thresh = value
            self.uncertainty_threshold_label.setText(f"{value:.2f}")
            self.uncertainty_threshold_slider.blockSignals(False)
    
    def _on_iou_changed(self, value):
        """Update slider/label when MainWindow changes"""
        if hasattr(self, 'iou_threshold_slider'):
            self.iou_threshold_slider.blockSignals(True)
            self.iou_threshold_slider.setValue(int(value * 100))
            self.iou_thresh = value
            self.iou_threshold_label.setText(f"{value:.2f}")
            self.iou_threshold_slider.blockSignals(False)
    
    def _on_area_changed(self, min_val, max_val):
        """Update sliders/label when MainWindow changes"""
        self.area_thresh_min = min_val
        self.area_thresh_max = max_val
        self._sync_area_widgets()
