"""
QtThresholdsWidget - Reusable widget for threshold controls
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QFormLayout, QLabel, QSlider, QSpinBox


class ThresholdsWidget(QGroupBox):
    """
    A reusable widget that provides threshold controls (max detections, uncertainty, IoU, area min/max).
    This widget can be configured to show only the controls needed for a specific use case.
    
    :param main_window: MainWindow object to sync threshold values
    :param show_max_detections: Whether to show the max detections spinbox
    :param show_uncertainty: Whether to show the uncertainty threshold slider
    :param show_iou: Whether to show the IoU threshold slider
    :param show_area: Whether to show the area threshold sliders (min and max)
    :param title: Title for the group box (default: "Thresholds")
    :param parent: Parent widget
    """
    
    def __init__(self, main_window, show_max_detections=False,
                 show_uncertainty=True, show_iou=False, show_area=False, 
                 title="Thresholds", parent=None):
        super().__init__(title, parent)
        
        self.main_window = main_window
        
        # Initialize threshold values from main window
        self.max_detections = main_window.get_max_detections()
        self.uncertainty_thresh = main_window.get_uncertainty_thresh()
        self.iou_thresh = main_window.get_iou_thresh()
        min_val, max_val = main_window.get_area_thresh()
        self.area_thresh_min = min_val
        self.area_thresh_max = max_val
        
        # Create the layout
        layout = QFormLayout()
        
        # Max detections spinbox
        if show_max_detections:
            self.max_detections_spinbox = QSpinBox()
            self.max_detections_spinbox.setRange(1, 10000)
            self.max_detections_spinbox.setValue(main_window.get_max_detections())
            self.max_detections_spinbox.valueChanged.connect(self._update_max_detections)
            main_window.maxDetectionsChanged.connect(self._on_max_detections_changed)
            layout.addRow("Max Detections:", self.max_detections_spinbox)
        
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
        
        self.setLayout(layout)
        
    def _update_max_detections(self, value):
        """Update max detections value"""
        self.max_detections = value
        self.main_window.update_max_detections(value)
    
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
