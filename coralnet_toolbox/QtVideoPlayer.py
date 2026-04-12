from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QCursor
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QSlider,
                             QLabel, QStyle, QSizePolicy, QStyleOptionSlider, QFrame, QApplication)

from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.Icons import get_icon


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
        # Call the base class to draw the standard slider, 
        # including the groove, default ticks, and handle.
        super().paintEvent(event)

        if not self._annotation_frames or (self.maximum() <= self.minimum()):
            return

        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        
        # Get the groove rectangle to position your custom marks
        groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        
        tick_color = QColor(230, 62, 0, 220)  # blood red
        painter.setPen(QPen(tick_color, 2))

        span = max(1, self.maximum() - self.minimum())
        for frame_idx in sorted(self._annotation_frames):
            ratio = (frame_idx - self.minimum()) / span
            x = groove.x() + int(ratio * groove.width())
            painter.drawLine(x, groove.top() - 2, x, groove.bottom() + 2)
        
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
    firstFrameClicked = pyqtSignal()
    lastFrameClicked = pyqtSignal()
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

        self.control_icon_tint = app_theme.TEXT_BRIGHT_COLOR

        self.first_frame_icon = get_icon("skip-to-start.svg", tint=self.control_icon_tint)
        self.prev_frame_icon = get_icon("rewind.svg", tint=self.control_icon_tint)
        self.play_icon = get_icon("play.svg", tint=self.control_icon_tint)
        self.pause_icon = get_icon("pause.svg", tint=self.control_icon_tint)
        self.stop_icon = get_icon("stop.svg", tint=self.control_icon_tint)
        self.next_frame_icon = get_icon("fast-forward.svg", tint=self.control_icon_tint)
        self.last_frame_icon = get_icon("skip-to-end.svg", tint=self.control_icon_tint)

        self.prev_annotated_icon = get_icon("left.svg", tint=self.control_icon_tint)
        self.next_annotated_icon = get_icon("right.svg", tint=self.control_icon_tint)
        
        self.setup_ui()

        # Prefer reuse of ImagePreviewTooltip from ImageWindow if available
        try:
            from coralnet_toolbox.QtImageWindow import ImagePreviewTooltip
            self.preview_tooltip = ImagePreviewTooltip(self)
        except Exception:
            self.preview_tooltip = FramePreviewTooltip(self)

        self._last_preview_idx = None
        self._is_user_scrubbing = False
        self._was_playing_before_scrub = False
        # Install event filter on slider to capture hover and clicks
        self.slider.installEventFilter(self)
        
    def setup_ui(self):
        """Initialize the layout and widgets."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5) # Small margins
        layout.setSpacing(10)

        # --- 0. Jump to First Frame Button ---
        self.btn_first = QPushButton()
        self.btn_first.setIcon(self.first_frame_icon)
        self.btn_first.setToolTip("Jump to First Frame")
        self.btn_first.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_first.setIconSize(app_theme.scale_size(18))
        self.btn_first.clicked.connect(self.jump_to_first_frame)
        layout.addWidget(self.btn_first)

        # --- 1. Step Backward Button ---
        self.btn_prev = QPushButton()
        self.btn_prev.setIcon(self.prev_frame_icon)
        self.btn_prev.setToolTip("Step Back 1 Frame")
        self.btn_prev.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_prev.setIconSize(app_theme.scale_size(18))
        self.btn_prev.clicked.connect(self.prevFrameClicked.emit)
        layout.addWidget(self.btn_prev)

        # --- 2. Play/Pause Button ---
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.play_icon)
        self.btn_play.setToolTip("Play / Pause")
        self.btn_play.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_play.setIconSize(app_theme.scale_size(18))
        self.btn_play.clicked.connect(self.toggle_playback_state)
        layout.addWidget(self.btn_play)

        # --- 3. Stop Button ---
        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(self.stop_icon)
        self.btn_stop.setToolTip("Stop Playback")
        self.btn_stop.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_stop.setIconSize(app_theme.scale_size(18))
        self.btn_stop.clicked.connect(self.stop_playback)
        layout.addWidget(self.btn_stop)

        # --- 4. Step Forward Button ---
        self.btn_next = QPushButton()
        self.btn_next.setIcon(self.next_frame_icon)
        self.btn_next.setToolTip("Step Forward 1 Frame")
        self.btn_next.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_next.setIconSize(app_theme.scale_size(18))
        self.btn_next.clicked.connect(self.nextFrameClicked.emit)
        layout.addWidget(self.btn_next)

        # --- 5. Jump to Last Frame Button ---
        self.btn_last = QPushButton()
        self.btn_last.setIcon(self.last_frame_icon)
        self.btn_last.setToolTip("Jump to Last Frame")
        self.btn_last.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_last.setIconSize(app_theme.scale_size(18))
        self.btn_last.clicked.connect(self.jump_to_last_frame)
        layout.addWidget(self.btn_last)

        # Add visual separator between step controls and scrubber
        separator = QFrame()
        separator.setStyleSheet("color: white;")
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # --- 6. Jump to Previous Annotated Frame Button ---
        self.btn_prev_annotated = QPushButton()
        self.btn_prev_annotated.setIcon(self.prev_annotated_icon)
        self.btn_prev_annotated.setToolTip("Jump to Previous Annotated Frame")
        self.btn_prev_annotated.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_prev_annotated.setIconSize(app_theme.scale_size(18))
        self.btn_prev_annotated.clicked.connect(self.prevAnnotatedClicked.emit)
        layout.addWidget(self.btn_prev_annotated)

        # --- 7. Jump to Next Annotated Frame Button ---
        self.btn_next_annotated = QPushButton()
        self.btn_next_annotated.setIcon(self.next_annotated_icon)
        self.btn_next_annotated.setToolTip("Jump to Next Annotated Frame")
        self.btn_next_annotated.setFixedSize(app_theme.scale_int(30), app_theme.scale_int(30))
        self.btn_next_annotated.setIconSize(app_theme.scale_size(18))
        self.btn_next_annotated.clicked.connect(self.nextAnnotatedClicked.emit)
        layout.addWidget(self.btn_next_annotated)

        # --- 8. Scrubber Slider (AnnotatedSlider draws per-frame annotation ticks) ---
        self.slider = AnnotatedSlider(Qt.Horizontal)
        self.slider.setToolTip("Seek Frame")
        self.slider.setMouseTracking(True)
        # During user scrubbing we will show thumbnails but delay the actual
        # seek until the user releases the slider to avoid loading frames.
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider)

        # --- 9. Frame Counter Label ---
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
                # Show preview when Ctrl is held or while the user is scrubbing (dragging)
                if (QApplication.keyboardModifiers() & Qt.ControlModifier) or self.slider.isSliderDown():
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
                        self._show_preview_for_frame_idx(frame_idx, event.globalPos())
                    return False
                else:
                    # Hide preview when neither Ctrl nor scrubbing
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

    def _on_slider_pressed(self):
        self._is_user_scrubbing = True
        self._last_preview_idx = None
        # If video was playing, pause it for scrubbing and remember state
        self._was_playing_before_scrub = bool(self.is_playing)
        if self._was_playing_before_scrub:
            self.set_paused()
            try:
                self.pauseClicked.emit()
            except Exception:
                pass

    def _on_slider_moved(self, value):
        # Show a thumbnail preview for the current drag position without
        # emitting a seek — the actual seek occurs on release.
        if value != self._last_preview_idx:
            self._last_preview_idx = value
            try:
                self._show_preview_for_frame_idx(value, QCursor.pos())
            except Exception:
                self.preview_tooltip.hide()

    def _on_slider_released(self):
        # User finished scrubbing — perform the actual seek and hide preview.
        self._is_user_scrubbing = False
        final_value = self.slider.value()
        self._last_preview_idx = None
        self.preview_tooltip.hide()
        self.seekChanged.emit(final_value)
        # Restore playback if it was playing before scrubbing
        if getattr(self, '_was_playing_before_scrub', False):
            # Ensure UI shows playing and notify controller
            self.set_playing()
            try:
                self.playClicked.emit()
            except Exception:
                pass
        self._was_playing_before_scrub = False

    def _show_preview_for_frame_idx(self, frame_idx, global_pos=None):
        # Resolve owning AnnotationWindow in case the player has been reparented
        aw = getattr(self, 'annotation_window', None)
        if aw is None:
            p = self.parent()
            while p is not None:
                if hasattr(p, '_active_video_raster'):
                    aw = p
                    break
                p = p.parent()

        if aw is None:
            self.preview_tooltip.hide()
            return

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
                    if isinstance(pix, QImage):
                        pm = QPixmap.fromImage(pix)
                    else:
                        pm = pix
                    self.preview_tooltip.set_image(pm)
                    if global_pos is None:
                        global_pos = QCursor.pos()
                    self.preview_tooltip.show_at(global_pos)
                else:
                    self.preview_tooltip.hide()
            else:
                self.preview_tooltip.hide()
        except Exception:
            self.preview_tooltip.hide()

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
        self.btn_play.setIcon(self.pause_icon)
        # Disable step/jump buttons while playing to prevent conflicts
        self.btn_first.setEnabled(False)
        self.btn_prev_annotated.setEnabled(False)
        self.btn_prev.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_next.setEnabled(False)
        self.btn_next_annotated.setEnabled(False)
        self.btn_last.setEnabled(False)

    def set_paused(self):
        """Force UI to 'Paused' state."""
        self.is_playing = False
        self.btn_play.setIcon(self.play_icon)
        self.btn_first.setEnabled(True)
        self.btn_prev_annotated.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.btn_next_annotated.setEnabled(True)
        self.btn_last.setEnabled(True)

    def stop_playback(self):
        """Stop playback and return the widget to a paused state."""
        self.set_paused()
        self.pauseClicked.emit()

    def jump_to_first_frame(self):
        """Jump to the first frame in the clip."""
        if self.total_frames <= 0:
            return
        self.seekChanged.emit(0)

    def jump_to_last_frame(self):
        """Jump to the last frame in the clip."""
        if self.total_frames <= 0:
            return
        self.seekChanged.emit(max(0, self.total_frames - 1))

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
        Update the tick marks on the scrub bar.

        Args:
            frame_indices (set[int]): Frame indices that contain at least one annotation.
        """
        self.slider.set_annotation_frames(frame_indices)

    def reset(self):
        """Reset controls to initial state."""
        self.set_paused()
        self.update_state(0, 0)
        self.slider.set_annotation_frames(set())