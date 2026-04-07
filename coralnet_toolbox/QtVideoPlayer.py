from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QSlider,
                             QLabel, QStyle, QSizePolicy, QStyleOptionSlider, QFrame, QApplication)


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

    def mousePressEvent(self, event):
        # Let the parent/VideoPlayerWidget handle Ctrl+click behavior via eventFilter.
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Forward to base implementation; VideoPlayerWidget installs an event filter
        # to handle Ctrl-hover previews.
        super().mouseMoveEvent(event)


class FramePreviewTooltip(QFrame):
    """
    Lightweight tooltip that shows a QPixmap. Provides `set_image(pixmap)` and
    `show_at(global_pos)` methods to mirror ImagePreviewTooltip API used elsewhere.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._pixmap = None

    def set_image(self, pixmap: QPixmap):
        if pixmap is None:
            self._pixmap = None
            self.hide()
            return
        self._pixmap = pixmap
        self._label.setPixmap(self._pixmap)
        self._label.adjustSize()
        self.adjustSize()

    def show_at(self, global_pos):
        if self._pixmap is None:
            return
        # Offset tooltip slightly above cursor
        geo = self._label.geometry()
        w = geo.width()
        h = geo.height()
        x = global_pos.x() - (w // 2)
        y = global_pos.y() - h - 20
        self.move(x, y)
        self.show()


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

    def __init__(self, parent=None, annotation_window=None):
        super().__init__(parent)
        # Direct reference to the owning AnnotationWindow (set by caller)
        self.annotation_window = annotation_window
        
        self.is_playing = False
        self.total_frames = 0
        
        self.setup_ui()

        # Prefer reuse of ImagePreviewTooltip from ImageWindow if available
        try:
            from coralnet_toolbox.QtImageWindow import ImagePreviewTooltip
            self.preview_tooltip = ImagePreviewTooltip(self)
        except Exception:
            self.preview_tooltip = FramePreviewTooltip(self)

        self._last_preview_idx = None
        # Install event filter on slider to capture hover and clicks
        self.slider.installEventFilter(self)
        
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
        self.slider.setMouseTracking(True)
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

    def eventFilter(self, obj, event):
        # Intercept events on the slider to implement Ctrl+hover preview and Ctrl+click seek
        if obj is self.slider:
            if event.type() == QEvent.MouseMove:
                # Show preview only when Ctrl is held
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    # Compute value from mouse x position
                    pos = event.pos()
                    w = max(1, self.slider.width())
                    x = pos.x()
                    ratio = min(max(0.0, x / w), 1.0)
                    vmin = self.slider.minimum()
                    vmax = self.slider.maximum()
                    if vmax > vmin:
                        frame_idx = int(round(vmin + ratio * (vmax - vmin)))
                    else:
                        frame_idx = vmin

                    if frame_idx != self._last_preview_idx:
                        self._last_preview_idx = frame_idx
                        # Resolve the AnnotationWindow that owns this player.  When the
                        # player is placed into a QToolBar it may be reparented, so
                        # prefer an explicit reference if provided.
                        aw = getattr(self, 'annotation_window', None)
                        if aw is None:
                            p = self.parent()
                            while p is not None:
                                if hasattr(p, '_active_video_raster'):
                                    aw = p
                                    break
                                p = p.parent()

                        raster = None
                        if aw is not None:
                            raster = getattr(aw, '_active_video_raster', None)

                        if aw is not None:
                            # Prefer RasterManager thumbnail API which handles videos
                            try:
                                rm = None
                                try:
                                    rm = aw.main_window.image_window.raster_manager
                                except Exception:
                                    rm = None

                                if rm is not None and hasattr(aw, '_active_video_raster') and aw._active_video_raster is not None:
                                    virtual = aw._active_video_raster.make_frame_path(aw._active_video_raster.image_path, frame_idx)
                                    pix = rm.get_thumbnail(virtual, longest_edge=256)
                                    if pix is not None:
                                        self.preview_tooltip.set_image(pix)
                                        self.preview_tooltip.show_at(event.globalPos())
                                    else:
                                        self.preview_tooltip.hide()
                                else:
                                    self.preview_tooltip.hide()
                            except Exception:
                                self.preview_tooltip.hide()
                        else:
                            self.preview_tooltip.hide()
                    return False
                else:
                    # Hide preview when Ctrl not held
                    self._last_preview_idx = None
                    self.preview_tooltip.hide()
                    return False

            if event.type() == QEvent.Leave:
                self._last_preview_idx = None
                self.preview_tooltip.hide()
                return False

            if event.type() == QEvent.MouseButtonPress:
                # Ctrl+click: jump to computed frame
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    mouse_event = event
                    if mouse_event.button() == Qt.LeftButton:
                        pos = mouse_event.pos()
                        w = max(1, self.slider.width())
                        x = pos.x()
                        ratio = min(max(0.0, x / w), 1.0)
                        vmin = self.slider.minimum()
                        vmax = self.slider.maximum()
                        if vmax > vmin:
                            frame_idx = int(round(vmin + ratio * (vmax - vmin)))
                        else:
                            frame_idx = vmin

                        # Update slider UI and emit seek
                        self.slider.setValue(frame_idx)
                        self.seekChanged.emit(frame_idx)
                        # Hide tooltip after selection
                        self._last_preview_idx = None
                        self.preview_tooltip.hide()
                        return True

        return super().eventFilter(obj, event)

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