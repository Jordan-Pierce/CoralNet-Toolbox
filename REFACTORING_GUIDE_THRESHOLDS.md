# Refactoring Guide: Using ThresholdsWidget

This guide shows how to refactor each deployment dialog to use the new `ThresholdsWidget` class from `coralnet_toolbox.Common`.

## Benefits
- **Eliminates code duplication**: All threshold UI code is in one place
- **Easier maintenance**: Bug fixes and improvements only need to be made once
- **Consistent behavior**: All dialogs work the same way
- **Flexible configuration**: Easy to show/hide specific thresholds per dialog

## Pattern for Refactoring

### Step 1: Update Imports

**Before:**
```python
from PyQt5.QtWidgets import (QApplication, QMessageBox, QLabel, QGroupBox, QFormLayout, QSlider, QSpinBox)
```

**After:**
```python
from PyQt5.QtWidgets import (QApplication, QMessageBox, QGroupBox, QFormLayout, QSpinBox)
from coralnet_toolbox.Common import ThresholdsWidget
```

### Step 2: Replace setup_thresholds_layout()

**Before:**
```python
def setup_thresholds_layout(self):
    """
    Setup threshold control section in a group box.
    """
    group_box = QGroupBox("Thresholds")
    layout = QFormLayout()

    # Uncertainty threshold controls
    self.uncertainty_thresh = self.main_window.get_uncertainty_thresh()
    self.uncertainty_threshold_slider = QSlider(Qt.Horizontal)
    # ... lots of slider setup code ...
    
    # IoU threshold controls
    self.iou_thresh = self.main_window.get_iou_thresh()
    self.iou_threshold_slider = QSlider(Qt.Horizontal)
    # ... lots of slider setup code ...
    
    # Area threshold controls
    # ... even more slider setup code ...
    
    group_box.setLayout(layout)
    self.layout.addWidget(group_box)
```

**After:**
```python
def setup_thresholds_layout(self):
    """
    Setup threshold control section using the reusable ThresholdsWidget.
    """
    # Create the thresholds widget with needed thresholds enabled
    self.thresholds_widget = ThresholdsWidget(
        self.main_window,
        show_uncertainty=True,
        show_iou=True,
        show_area=True
    )
    self.layout.addWidget(self.thresholds_widget)
```

### Step 3: Update showEvent()

**Before:**
```python
def showEvent(self, event):
    super().showEvent(event)
    self.initialize_uncertainty_threshold()
    self.initialize_iou_threshold()
    self.initialize_area_threshold()
```

**After:**
```python
def showEvent(self, event):
    super().showEvent(event)
    # Initialize thresholds in the widget
    self.thresholds_widget.initialize_thresholds()
```

### Step 4: Remove Old Initialization Methods from Base Class

You can remove these methods from `QtBase.py` since they're now handled by the widget:
- `initialize_uncertainty_threshold()`
- `initialize_iou_threshold()`
- `initialize_area_threshold()`
- `update_uncertainty_label()`
- `update_iou_label()`
- `update_area_label()`

## Configuration for Each Dialog

### QtClassify.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=False,
    show_uncertainty=True,
    show_iou=False,
    show_area=False
)
```

### QtDetect.py & QtSegment.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=True,
    max_detections_value=300,  # or self.max_detect
    show_uncertainty=True,
    show_iou=True,
    show_area=True
)
```

### QtSemantic.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=False,
    show_uncertainty=True,
    show_iou=False,
    show_area=False
)
```

### SAM/QtDeployGenerator.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=True,
    max_detections_value=self.max_detect,
    show_uncertainty=True,
    show_iou=True,
    show_area=True
)
self.left_panel.addWidget(self.thresholds_widget)  # Note: uses left_panel instead of layout
```

### SeeAnything/QtDeployGenerator.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=True,
    max_detections_value=self.max_detect,
    show_uncertainty=True,
    show_iou=True,
    show_area=True
)
self.left_panel.addWidget(self.thresholds_widget)  # Note: uses left_panel instead of layout
```

### Transformers/QtDeployModel.py
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=False,
    show_uncertainty=True,
    show_iou=True,
    show_area=True
)
```

## Additional Notes

### Accessing Max Detections Value
When you need to access the max detections value in your code (e.g., in the `predict()` method):

**Before (with separate spinbox):**
```python
max_det=self.max_detections_spinbox.value()
```

**After (with ThresholdsWidget):**
```python
max_det=self.thresholds_widget.max_detections_spinbox.value()
```

Or if max_detections is not shown in the widget, you can still use the old pattern by keeping a separate spinbox in `setup_parameters_layout()`.

### For SAM and SeeAnything QtDeployPredictor dialogs
These dialogs also have threshold controls but they're structured slightly differently. They don't inherit from Base and have their own layout structure. You can still use ThresholdsWidget but may need to adjust how it's added to the layout.

### Custom Title
You can customize the group box title:
```python
self.thresholds_widget = ThresholdsWidget(
    self.main_window,
    show_max_detections=True,
    max_detections_value=300,
    show_uncertainty=True,
    show_iou=True,
    show_area=True,
    title="Detection Thresholds"  # Custom title
)
```

### Testing After Refactoring
1. Load each dialog and verify all threshold sliders appear correctly
2. Adjust sliders and verify values update in main window
3. Close and reopen dialogs to verify initialization works
4. Run predictions to ensure threshold values are being used correctly

## Benefits Recap
- **~60 lines of code** reduced to **~7 lines** per dialog
- **Consistent behavior** across all dialogs
- **Single source of truth** for threshold UI logic
- **Easier to add new features** (e.g., tooltips, validation)
