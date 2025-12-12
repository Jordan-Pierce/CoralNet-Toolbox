from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QSlider, 
                             QLabel, QStyle, QSizePolicy)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


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

        # --- 1. Step Backward Button ---
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.btn_prev.setToolTip("Step Back 1 Frame")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(self.prevFrameClicked.emit)
        layout.addWidget(self.btn_prev)

        # --- 2. Play/Pause Button ---
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play / Pause (Spacebar)")
        self.btn_play.setFixedWidth(30)
        self.btn_play.clicked.connect(self.toggle_playback_state)
        layout.addWidget(self.btn_play)

        # --- 3. Step Forward Button ---
        self.btn_next = QPushButton()
        self.btn_next.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.btn_next.setToolTip("Step Forward 1 Frame")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(self.nextFrameClicked.emit)
        layout.addWidget(self.btn_next)

        # --- 4. Scrubber Slider ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setToolTip("Seek Frame")
        # Connect sliderReleased/valueChanged depending on desired behavior.
        # using valueChanged allows dragging to scrub, but might be heavy if not optimized.
        # We generally use sliderMoved for scrubbing and valueChanged for clicks.
        self.slider.sliderMoved.connect(self.seekChanged.emit) 
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider)

        # --- 5. Frame Counter Label ---
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
        # Disable step buttons while playing to prevent conflicts
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

    def set_paused(self):
        """Force UI to 'Paused' state."""
        self.is_playing = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

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

    def reset(self):
        """Reset controls to initial state."""
        self.set_paused()
        self.update_state(0, 0)