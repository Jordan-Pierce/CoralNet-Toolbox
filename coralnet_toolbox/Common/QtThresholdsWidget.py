"""
QtThresholdsWidget - Reusable widget for threshold controls
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QGroupBox, QFormLayout, QLabel, QSlider, QSpinBox


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
        
        # Create the layout
        layout = QFormLayout()

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
            self.area_threshold_min_slider = QSlider(Qt.Horizontal)
            self.area_threshold_min_slider.setRange(0, 100)
            self.area_threshold_min_slider.setValue(int(self.area_thresh_min * 100))
            self.area_threshold_min_slider.setTickPosition(QSlider.TicksBelow)
            self.area_threshold_min_slider.setTickInterval(10)
            self.area_threshold_min_slider.valueChanged.connect(self._update_area_label)
            
            self.area_threshold_max_slider = QSlider(Qt.Horizontal)
            self.area_threshold_max_slider.setRange(0, 100)
            self.area_threshold_max_slider.setValue(int(self.area_thresh_max * 100))
            self.area_threshold_max_slider.setTickPosition(QSlider.TicksBelow)
            self.area_threshold_max_slider.setTickInterval(10)
            self.area_threshold_max_slider.valueChanged.connect(self._update_area_label)
            
            main_window.areaChanged.connect(self._on_area_changed)
            
            self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
            layout.addRow("Area Threshold Min", self.area_threshold_min_slider)
            layout.addRow("Area Threshold Max", self.area_threshold_max_slider)
            layout.addRow("", self.area_threshold_label)
            apply_row_tooltip(
                self.area_threshold_min_slider,
                "Lower bound of the normalized annotation area filter. Objects smaller than this "
                "fraction of the image area are removed after confidence and IoU filtering.")
            apply_row_tooltip(
                self.area_threshold_max_slider,
                "Upper bound of the normalized annotation area filter. Objects larger than this "
                "fraction of the image area are removed after confidence and IoU filtering.")
            self.area_threshold_label.setToolTip(
                "Current area filter range, shown as normalized fractions of the image area.")
        
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
    
    def _update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val = self.area_threshold_min_slider.value()
        max_val = self.area_threshold_max_slider.value()
        if min_val > max_val:
            min_val = max_val
            self.area_threshold_min_slider.setValue(min_val)
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
    
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
            current_min, current_max = self.main_window.get_area_thresh()
            self.area_threshold_min_slider.setValue(int(current_min * 100))
            self.area_threshold_max_slider.setValue(int(current_max * 100))
            self.area_thresh_min = current_min
            self.area_thresh_max = current_max
    
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
        if hasattr(self, 'area_threshold_min_slider') and hasattr(self, 'area_threshold_max_slider'):
            self.area_threshold_min_slider.blockSignals(True)
            self.area_threshold_max_slider.blockSignals(True)
            self.area_threshold_min_slider.setValue(int(min_val * 100))
            self.area_threshold_max_slider.setValue(int(max_val * 100))
            self.area_thresh_min = min_val
            self.area_thresh_max = max_val
            self.area_threshold_label.setText(f"{min_val:.2f} - {max_val:.2f}")
            self.area_threshold_min_slider.blockSignals(False)
            self.area_threshold_max_slider.blockSignals(False)
