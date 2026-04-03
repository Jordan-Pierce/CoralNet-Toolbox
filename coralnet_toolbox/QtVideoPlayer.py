from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QSlider,
                             QLabel, QStyle, QSizePolicy, QStyleOptionSlider)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotatedSlider(QSlider):
    """
    A QSlider subclass that draws small tick marks at frame positions that
    contain annotations, giving a visual overview of annotated content.
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._annotation_frames: set = set()

    def set_annotation_frames(self, frames):
        """Update the set of frame indices that have annotations and repaint."""
        self._annotation_frames = set(frames)
        self.update()

    def paintEvent(self, event):
        # Draw the standard slider first
        super().paintEvent(event)

        if not self._annotation_frames or self.maximum() <= 0:
            return

        # Use the style to find the exact groove rectangle
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        painter = QPainter(self)
        tick_color = QColor(255, 190, 0, 220)   # amber / gold
        pen = QPen(tick_color, 2)
        painter.setPen(pen)

        span = self.maximum() - self.minimum()
        gx = groove.x()
        gw = groove.width()
        gy_top = groove.y()
        gy_bot = groove.y() + groove.height()

        for frame_idx in self._annotation_frames:
            ratio = (frame_idx - self.minimum()) / span
            x = gx + int(ratio * gw)
            # Draw a short line that straddles the groove edges
            painter.drawLine(x, gy_top - 2, x, gy_bot + 2)

        painter.end()


class VideoPlayerWidget(QWidget):
    """
    A widget containing video playback controls: Play/Pause, Seek Slider, 
    Step Forward/Back, and a Frame Counter.
    """
    
    # Signals to notify the parent (AnnotationWindow) of user actions
    playClicked = pyqtSignal()
    pauseClicked = pyqtSignal()
    seekChanged = pyqtSignal(int)       # Emits the new frame index
    nextFrameClicked = pyqtSignal()
    prevFrameClicked = pyqtSignal()
    nextAnnotatedClicked = pyqtSignal()
    prevAnnotatedClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_playing = False
        self.total_frames = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the layout and widgets."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5) # Small margins
        layout.setSpacing(10)

        # --- 1. Jump to Previous Annotated Frame Button ---
        self.btn_prev_annotated = QPushButton()
        self.btn_prev_annotated.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_prev_annotated.setToolTip("Jump to Previous Annotated Frame")
        self.btn_prev_annotated.setFixedWidth(30)
        self.btn_prev_annotated.clicked.connect(self.prevAnnotatedClicked.emit)
        layout.addWidget(self.btn_prev_annotated)

        # --- 2. Step Backward Button ---
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.btn_prev.setToolTip("Step Back 1 Frame")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(self.prevFrameClicked.emit)
        layout.addWidget(self.btn_prev)

        # --- 3. Play/Pause Button ---
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.setFixedWidth(30)
        self.btn_play.clicked.connect(self.toggle_playback_state)
        layout.addWidget(self.btn_play)

        # --- 4. Step Forward Button ---
        self.btn_next = QPushButton()
        self.btn_next.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.btn_next.setToolTip("Step Forward 1 Frame")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(self.nextFrameClicked.emit)
        layout.addWidget(self.btn_next)

        # --- 5. Jump to Next Annotated Frame Button ---
        self.btn_next_annotated = QPushButton()
        self.btn_next_annotated.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        self.btn_next_annotated.setToolTip("Jump to Next Annotated Frame")
        self.btn_next_annotated.setFixedWidth(30)
        self.btn_next_annotated.clicked.connect(self.nextAnnotatedClicked.emit)
        layout.addWidget(self.btn_next_annotated)

        # --- 6. Scrubber Slider (AnnotatedSlider draws per-frame annotation ticks) ---
        self.slider = AnnotatedSlider(Qt.Horizontal)
        self.slider.setToolTip("Seek Frame")
        # Connect sliderReleased/valueChanged depending on desired behavior.
        # using valueChanged allows dragging to scrub, but might be heavy if not optimized.
        # We generally use sliderMoved for scrubbing and valueChanged for clicks.
        self.slider.sliderMoved.connect(self.seekChanged.emit) 
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider)

        # --- 7. Frame Counter Label ---
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setToolTip("Current Frame / Total Frames")
        self.lbl_frame.setMinimumWidth(80)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_frame)

    def _on_slider_value_changed(self, value):
        """
        Handle clicks on the slider bar (jump to position).
        We distinguish between programmatic updates (blockSignals) and user clicks.
        """
        # Only emit if the slider is not currently being dragged (handled by sliderMoved)
        if not self.slider.isSliderDown():
            self.seekChanged.emit(value)

    def toggle_playback_state(self):
        """
        Toggles the internal play state and updates the icon.
        Emits the appropriate signal for the controller to handle the logic.
        """
        if self.is_playing:
            self.set_paused()
            self.pauseClicked.emit()
        else:
            self.set_playing()
            self.playClicked.emit()

    def set_playing(self):
        """Force UI to 'Playing' state."""
        self.is_playing = True
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        # Disable step/jump buttons while playing to prevent conflicts
        self.btn_prev_annotated.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_next_annotated.setEnabled(False)

    def set_paused(self):
        """Force UI to 'Paused' state."""
        self.is_playing = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_prev_annotated.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_next_annotated.setEnabled(True)

    def update_state(self, current_frame, total_frames):
        """
        Update the slider and label from the external source (AnnotationWindow).
        Uses blockSignals to prevent creating a feedback loop.
        """
        self.total_frames = total_frames
        
        # Update Slider
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, total_frames - 1))
        self.slider.setValue(current_frame)
        self.slider.blockSignals(False)
        
        # Update Label
        self.lbl_frame.setText(f"{current_frame} / {total_frames}")

    def update_annotation_marks(self, frame_indices):
        """
        Update the amber tick marks on the scrub bar.

        Args:
            frame_indices (set[int]): Frame indices that contain at least one annotation.
        """
        self.slider.set_annotation_frames(frame_indices)

    def reset(self):
        """Reset controls to initial state."""
        self.set_paused()
        self.update_state(0, 0)
        self.slider.set_annotation_frames(set())