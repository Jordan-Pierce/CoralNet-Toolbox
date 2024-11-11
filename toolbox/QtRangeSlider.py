from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QSlider, QStyleOptionSlider, QStyle, QSpinBox, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QRect


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class QRangeSlider(QSlider):
    # Signal emits min and max values when either changes
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create layout
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create widgets
        self.min_spin = QSpinBox()
        self.slider = QRangeSlider()
        self.max_spin = QSpinBox()
        
        # Setup spin boxes
        self.min_spin.setRange(self.slider.minimum(), self.slider.maximum())
        self.max_spin.setRange(self.slider.minimum(), self.slider.maximum())
        
        # Add widgets to layout
        layout.addWidget(self.min_spin)
        layout.addWidget(self.slider)
        layout.addWidget(self.max_spin)
        self.setLayout(layout)
        
        # Connect signals
        self.slider.rangeChanged.connect(self._update_spin_boxes)
        self.min_spin.valueChanged.connect(self._min_spin_changed)
        self.max_spin.valueChanged.connect(self._max_spin_changed)
        
        # Initial values
        min_val, max_val = self.slider.value()
        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)
        
    def _update_spin_boxes(self, min_val, max_val):
        """Update spin boxes when slider changes"""
        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)

    def _min_spin_changed(self, value):
        """Update slider when min spin box changes"""
        current_min, current_max = self.slider.value()
        self.slider.setValue((value, current_max))

    def _max_spin_changed(self, value):
        """Update slider when max spin box changes"""
        current_min, current_max = self.slider.value()
        self.slider.setValue((current_min, value))

    def setRange(self, min_value, max_value):
        """Set the range while ensuring min <= max"""
        self._min = min(min_value, max_value)
        self._max = max(min_value, max_value)
        self.rangeChanged.emit(self._min, self._max)
        self.update()
        
    def value(self):
        """Get current range values"""
        return self.slider.value()

    def setValue(self, value):
        """Set range values"""
        self.slider.setValue(value)

    def _pixelPosToValue(self, pos):
        """Convert pixel position to slider value"""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        span = self.maximum() - self.minimum()
        if span <= 0:
            return self.minimum()

        slider_length = groove.width()
        slider_min = groove.x()
        slider_max = groove.right() - handle.width()

        if pos < slider_min:
            return self.minimum()
        if pos > slider_max:
            return self.maximum()

        offset = pos - slider_min
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), offset, slider_length - handle.width())

    def mousePressEvent(self, event):
        """Handle mouse press to determine which slider to move"""
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

            # Calculate handle positions
            min_pos = self._valueToPixelPos(self._min)
            max_pos = self._valueToPixelPos(self._max)

            # Create handle rectangles
            min_handle = QRect(min_pos - self._handle_width // 2, handle.y(),
                               self._handle_width, handle.height())
            max_handle = QRect(max_pos - self._handle_width // 2, handle.y(),
                               self._handle_width, handle.height())

            # Determine which handle was clicked
            pos = event.pos()
            if min_handle.contains(pos):
                self._active_slider = 'min'
            elif max_handle.contains(pos):
                self._active_slider = 'max'
            else:
                # Click was on the slider track - move closest handle
                if abs(pos.x() - min_pos) < abs(pos.x() - max_pos):
                    self._active_slider = 'min'
                else:
                    self._active_slider = 'max'

            # Move the selected handle
            if self._active_slider:
                value = self._pixelPosToValue(pos.x())
                self._updateValue(value)
                event.accept()

    def mouseMoveEvent(self, event):
        """Update slider position during drag"""
        if event.buttons() & Qt.LeftButton and self._active_slider:
            value = self._pixelPosToValue(event.pos().x())
            self._updateValue(value)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Reset active slider on mouse release"""
        if event.button() == Qt.LeftButton:
            self._active_slider = None
            event.accept()

    def _updateValue(self, value):
        """Update either min or max value while maintaining order"""
        if self._active_slider == 'min':
            self._min = min(value, self._max)
        else:
            self._max = max(value, self._min)
        self.rangeChanged.emit(self._min, self._max)
        self.update()

    def _valueToPixelPos(self, value):
        """Convert slider value to pixel position"""
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        span = self.maximum() - self.minimum()
        if span <= 0:
            return groove.x()

        slider_length = groove.width() - handle.width()
        slider_pos = QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), value, slider_length)
        return groove.x() + slider_pos + handle.width() // 2

    def paintEvent(self, event):
        """Custom painting for the range slider"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get basic dimensions
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        # Draw the groove
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._inactive_color)
        painter.drawRect(groove)

        # Draw the selected range
        min_pos = self._valueToPixelPos(self._min)
        max_pos = self._valueToPixelPos(self._max)

        # Draw range rectangle
        range_rect = QRect(
            min_pos - self._handle_width // 2,
            groove.y(),
            max_pos - min_pos + self._handle_width,
            groove.height()
        )
        painter.setBrush(self._range_color)
        painter.drawRect(range_rect)

        # Draw handles
        painter.setBrush(self._active_color)

        # Min handle
        min_handle = QRect(
            min_pos - self._handle_width // 2,
            handle.y(),
            self._handle_width,
            handle.height()
        )
        painter.drawRoundedRect(min_handle, 2, 2)

        # Max handle
        max_handle = QRect(
            max_pos - self._handle_width // 2,
            handle.y(),
            self._handle_width,
            handle.height()
        )
        painter.drawRoundedRect(max_handle, 2, 2)

    def value(self):
        """Return current min and max values"""
        return self._min, self._max

    def setValue(self, value):
        """Set slider values"""
        if isinstance(value, tuple):
            self._min, self._max = min(value), max(value)
        else:
            # Single value - move closest handle
            if self._active_slider == 'min':
                self._min = min(value, self._max)
            elif self._active_slider == 'max':
                self._max = max(value, self._min)
            else:
                if abs(value - self._min) < abs(value - self._max):
                    self._min = value
                else:
                    self._max = value
        self.rangeChanged.emit(self._min, self._max)
        self.update()