import os 
from datetime import datetime

import cv2
import numpy as np
import supervision as sv

from shapely.geometry import Polygon, Point

from ultralytics import YOLO

from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, 
                             QWidget, QListWidget, QListWidgetItem, QFrame,
                             QAbstractItemView, QFormLayout, QComboBox, QSizePolicy,
                             QMessageBox, QApplication)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoDisplayWidget(QWidget):
    """Custom widget for displaying video frames and handling mouse events for region drawing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Paint the video frame and regions on this widget."""
        if not self.parent_widget:
            return
            
        painter = QPainter(self)
        
        if self.parent_widget.current_frame is not None:
            # Convert frame to QImage and QPixmap
            rgb = cv2.cvtColor(self.parent_widget.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale while maintaining aspect ratio to fit this widget
            widget_width = self.width()
            widget_height = self.height()
            scaled = pixmap.scaled(widget_width, widget_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Center the scaled image within this widget
            offset_x = (widget_width - scaled.width()) // 2
            offset_y = (widget_height - scaled.height()) // 2
            
            painter.drawPixmap(offset_x, offset_y, scaled)
            
        # Draw rectangles and polygons (these are in widget coordinates)
        if self.parent_widget.show_regions:
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            for region in self.parent_widget.regions:
                if isinstance(region, dict) and region.get("type") == "polygon":
                    points = region["points"]
                    if len(points) > 1:
                        for i in range(len(points)):
                            painter.drawLine(points[i], points[(i + 1) % len(points)])
                else:
                    # Backward compatibility: treat as QRect or dict with type 'rect'
                    rect = region["rect"] if isinstance(region, dict) and region.get("type") == "rect" else region
                    painter.drawRect(rect)
                    
        # Draw current rectangle being drawn
        if self.parent_widget.drawing and self.parent_widget.current_rect:
            pen = QPen(Qt.green, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.parent_widget.current_rect)
            
        # Draw in-progress polygon
        if self.parent_widget.drawing_polygon and self.parent_widget.current_polygon_points:
            pen = QPen(Qt.green, 2, Qt.DashLine)
            painter.setPen(pen)
            pts = self.parent_widget.current_polygon_points
            for i in range(1, len(pts)):
                painter.drawLine(pts[i - 1], pts[i])
            
    def mousePressEvent(self, event):
        """Forward mouse press events to parent widget with proper coordinates."""
        if self.parent_widget:
            self.parent_widget.mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Forward mouse move events to parent widget with proper coordinates."""
        if self.parent_widget:
            self.parent_widget.mouseMoveEvent(event)


class VideoRegionWidget(QWidget):
    """Widget for displaying video, playback controls, and drawing/editing rectangular regions only."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent
        
        # Polygon drawing state (must be set before any UI or event setup)
        self.drawing_polygon = False
        self.current_polygon_points = []

        # Inference
        self.inference_engine = InferenceEngine(self)
        self.inference_enabled = False
        self.region_polygons = []

        # Video frame and display
        self.frame = None
        self.pixmap = None
        self.regions = []  # List of dicts: {"type": "rect"/"polygon", ...}
        self.drawing = False
        self.rect_start = None  # QPoint
        self.rect_end = None    # QPoint
        self.current_rect = None  # QRect
        self.selected_region = None
        self.setMinimumSize(640, 480)

        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []

        # Region visibility
        self.show_regions = True

        # Video playback attributes
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.current_frame = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_speed = 1.0

        # Video output handling - simplified and fixed
        self.video_sink = None
        self.video_path = None
        self.output_dir = None
        self.output_path = None
        self.should_write_video = False

        self._setup_ui()
        self.disable_video_region()
        self.setFocusPolicy(Qt.StrongFocus)  # Needed for key events

    def _setup_ui(self):
        """Setup the user interface components."""
        # Main layout for the widget
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # Video Player GroupBox - takes up most of the space
        self.video_group = QGroupBox("Video Player")
        self.video_layout = QVBoxLayout(self.video_group)
        self.video_layout.setContentsMargins(5, 5, 5, 5)
        
        # Video display area using custom widget
        self.video_display = VideoDisplayWidget(self)
        self.video_layout.addWidget(self.video_display)
        
        # Add video group to main layout with stretch factor
        self.layout.addWidget(self.video_group, stretch=1)

        # Media Controls GroupBox - fixed height
        self.controls_group = QGroupBox("Media Controls")
        self.controls_group.setMaximumHeight(100)
        self.controls_group.setMinimumHeight(100)
        controls = QHBoxLayout(self.controls_group)
        controls.setContentsMargins(10, 10, 10, 10)

        # Main playback controls
        self.step_back_btn = QPushButton()
        self.step_back_btn.setIcon(self.style().standardIcon(self.style().SP_MediaSeekBackward))
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.step_back_btn)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.play_btn.setToolTip("Play")
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.play_btn)

        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPause))
        self.pause_btn.setToolTip("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.pause_btn)

        self.step_fwd_btn = QPushButton()
        self.step_fwd_btn.setIcon(self.style().standardIcon(self.style().SP_MediaSeekForward))
        self.step_fwd_btn.setToolTip("Step Forward")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        self.step_fwd_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.step_fwd_btn)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_btn.setToolTip("Stop & Reset")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.stop_btn)

        # Add vertical line separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        controls.addWidget(separator)

        # Record Play and Stop buttons (no groupbox)
        self.record_play_btn = QPushButton()
        self.record_play_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.record_play_btn.setToolTip("Start Recording")
        self.record_play_btn.setFocusPolicy(Qt.NoFocus)
        self.record_play_btn.clicked.connect(self.start_recording)
        self.record_play_btn.setEnabled(False)  # Only enabled if output_dir is set
        controls.addWidget(self.record_play_btn)
        self.record_stop_btn = QPushButton()
        self.record_stop_btn.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.record_stop_btn.setToolTip("Stop Recording")
        self.record_stop_btn.setFocusPolicy(Qt.NoFocus)
        self.record_stop_btn.clicked.connect(self.stop_recording)
        self.record_stop_btn.setEnabled(False)
        controls.addWidget(self.record_stop_btn)

        # Seek slider
        controls.addSpacing(8)
        controls.addStretch(1)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek)
        self.seek_slider.setMinimumWidth(300)
        controls.addWidget(self.seek_slider)
        controls.addStretch(1)
        controls.addSpacing(8)

        # Set button sizes
        max_btn_size = 32
        for btn in [self.step_back_btn, 
                    self.play_btn,
                    self.pause_btn, 
                    self.step_fwd_btn, 
                    self.stop_btn, 
                    self.record_play_btn, 
                    self.record_stop_btn]:
            btn.setMaximumSize(max_btn_size, max_btn_size)

        # Frame label and speed control
        self.frame_label = QLabel("Frame: 0 / 0")
        controls.addWidget(self.frame_label)

        self.speed_dropdown = QComboBox()
        self.speed_dropdown.addItems(["0.5x", "1x", "2x"])
        self.speed_dropdown.setCurrentIndex(1)
        self.speed_dropdown.currentIndexChanged.connect(self.change_speed)
        self.speed_dropdown.setMaximumWidth(80)
        controls.addWidget(self.speed_dropdown)

        # Add controls group to main layout with no stretch
        self.layout.addWidget(self.controls_group, stretch=0)
        
    def closeEvent(self, event):
        """Handle widget close event."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
        
    def enable_video_region(self):
        """Enable all controls in the video region widget."""
        self.setEnabled(True)
        for widget in [self.step_back_btn, 
                       self.play_btn, 
                       self.pause_btn, 
                       self.step_fwd_btn, 
                       self.stop_btn, 
                       self.seek_slider, 
                       self.speed_dropdown, 
                       self.frame_label]:
            widget.setEnabled(True)
            
        for widget in [self.record_play_btn, self.record_stop_btn]:
            # Enable record buttons only if output directory is set
            widget.setEnabled(bool(self.output_dir and os.path.exists(self.output_dir) and self.video_path))

    def disable_video_region(self):
        """Disable all controls in the video region widget."""
        self.setEnabled(False)
        for widget in [self.step_back_btn, 
                       self.play_btn, 
                       self.pause_btn, 
                       self.record_play_btn,
                       self.record_stop_btn,
                       self.step_fwd_btn, 
                       self.stop_btn, 
                       self.seek_slider, 
                       self.speed_dropdown, 
                       self.frame_label]:
            widget.setEnabled(False)
            
    def play_video(self):
        """Play the video from the current position."""
        if not self.is_playing and self.cap:
            self.is_playing = True
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)

    def pause_video(self):
        """Pause the video playback."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)

    def stop_video(self):
        """Stop the video playback and finalize output."""
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            
        # Finalize current video output when stopped
        self.stop_recording()
        print("Video recording stopped - ready for new recording on next play")
        
        # Reset to first frame
        if self.cap:
            self.seek(0)
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.seek_slider.setValue(0)
            self.update_frame_label()
        
        # Clear regions and reset state
        self.clear_regions()
        
        # Disable Inference from parent widget
        self.parent.disable_inference()
        
    def start_recording(self):
        """Start recording the video to output file. Also start playback if not already playing."""
        if not self.should_write_video and self.output_dir and os.path.exists(self.output_dir) and self.video_path:
            self._setup_video_output(self.video_path, self.output_dir)
        if not self.is_playing:
            self.play_video()
        self.update_record_buttons()

    def stop_recording(self):
        """Stop recording the video and finalize output."""
        self._cleanup_video_sink()
        self.update_record_buttons()
        
    def update_record_buttons(self):
        """Update the enabled state of record buttons based on output directory and recording status."""
        # Record Play button should be enabled if an output directory is set,
        # a video is loaded, and we are not currently recording.
        can_start_recording = bool(self.output_dir and os.path.exists(self.output_dir) and self.video_path)
        
        self.record_play_btn.setEnabled(can_start_recording and not self.should_write_video)
        
        # Record Stop button should be enabled only if we are currently recording.
        self.record_stop_btn.setEnabled(self.should_write_video)

    def seek(self, frame_number):
        """Seek to a specific frame in the video."""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            # Process the frame for inference if enabled
            processed_frame = self.process_frame_for_inference(frame.copy())
            
            self.current_frame = processed_frame
            self.current_frame_number = frame_number
            self.video_display.update()  # Update the video display widget
            self.update_frame_label()

    def step_forward(self):
        """Step forward one frame in the video."""
        if self.cap:
            next_frame = min(self.current_frame_number + 1, self.total_frames - 1)
            self.seek(next_frame)

    def step_backward(self):
        """Step backward one frame in the video."""
        if self.cap:
            prev_frame = max(self.current_frame_number - 1, 0)
            self.seek(prev_frame)

    def change_speed(self, idx):
        """Change the playback speed based on the selected index."""
        speeds = [0.5, 1.0, 2.0]
        self.playback_speed = speeds[idx]
        if self.is_playing:
            # Restart timer with new speed
            self.timer.stop()
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))

    def update_frame_label(self):
        """Update the frame label with current frame number and total frames."""
        self.frame_label.setText(f"Frame: {self.current_frame_number + 1} / {self.total_frames}")
        
    def enable_inference(self, enable: bool):
        """Enable or disable inference in the video region."""
        self.inference_enabled = enable

    def set_region_visibility(self, visible: bool):
        """Set the visibility of regions in the video."""
        self.show_regions = visible
        self.video_display.update()
        
    def clear_regions(self):
        """Clear all regions and reset the region polygons."""
        self.regions.clear()
        self.update_region_polygons()
        self.video_display.update()
        
        # Redraw the current frame without region overlays
        self.seek(self.current_frame_number)
        
    def update_region_polygons(self):
        """Update region polygons (QRects to shapely Polygons), mapping from widget to video frame coordinates."""
        self.region_polygons = []
        if self.current_frame is None or not self.regions:
            return
            
        # Get video frame dimensions
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # Get video display area dimensions
        widget_w = self.video_display.width()
        widget_h = self.video_display.height()
        
        # Calculate scaling and positioning (same logic as paintEvent in VideoDisplayWidget)
        scale = min(widget_w / frame_w, widget_h / frame_h)
        scaled_w = int(frame_w * scale)
        scaled_h = int(frame_h * scale)
        offset_x = (widget_w - scaled_w) // 2
        offset_y = (widget_h - scaled_h) // 2
        
        for region in self.regions:
            if isinstance(region, dict) and region.get("type") == "rect":
                rect = region["rect"]
                # Map QRect from video display widget coordinates to video frame coordinates
                # First, subtract the offset to get coordinates relative to the scaled video
                left_scaled = rect.left() - offset_x
                top_scaled = rect.top() - offset_y
                right_scaled = rect.right() - offset_x
                bottom_scaled = rect.bottom() - offset_y
                
                # Then scale back to original video frame coordinates
                left_frame = left_scaled / scale
                top_frame = top_scaled / scale
                right_frame = right_scaled / scale
                bottom_frame = bottom_scaled / scale
                
                # Clamp to frame bounds and ensure valid rectangle
                left_frame = max(0, min(left_frame, frame_w))
                right_frame = max(0, min(right_frame, frame_w))
                top_frame = max(0, min(top_frame, frame_h))
                bottom_frame = max(0, min(bottom_frame, frame_h))
                
                # Ensure we have a valid rectangle after clamping
                if left_frame < right_frame and top_frame < bottom_frame:
                    # Create polygon in video frame coordinates
                    poly = Polygon([
                        (left_frame, top_frame),
                        (right_frame, top_frame),
                        (right_frame, bottom_frame),
                        (left_frame, bottom_frame)
                    ])
                    self.region_polygons.append(poly)
            elif isinstance(region, dict) and region.get("type") == "polygon":
                pts = []
                for pt in region["points"]:
                    x_scaled = (pt.x() - offset_x) / scale
                    y_scaled = (pt.y() - offset_y) / scale
                    x_scaled = max(0, min(x_scaled, frame_w))
                    y_scaled = max(0, min(y_scaled, frame_h))
                    pts.append((x_scaled, y_scaled))
                if len(pts) > 2:
                    self.region_polygons.append(Polygon(pts))

    def mousePressEvent(self, event):
        """Handle mouse press events for drawing regions."""
        # Pause video playback if currently playing
        if self.is_playing:
            self.pause_video()
            
        if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
            # Polygon drawing mode
            pos = event.pos()
            if not self.drawing_polygon:
                self.drawing_polygon = True
                self.current_polygon_points = [pos, pos]  # Add first point and a preview point
            else:
                self.current_polygon_points.insert(-1, pos)  # Insert before the preview point
            # Update the video display to show the polygon in progress
            self.video_display.update()
            
        elif event.button() == Qt.LeftButton and not (event.modifiers() & Qt.ControlModifier):
            # Rectangle drawing as before
            # Get the current mouse position
            pos = event.pos()
            # Check if we're not currently drawing a rectangle
            if not self.drawing:
                # Start drawing a new rectangle
                self.drawing = True
                # Set the starting point of the rectangle
                self.rect_start = pos
                # Set the ending point to the same position initially
                self.rect_end = pos
                # Clear any current rectangle being drawn
                self.current_rect = None
            else:
                # Finish drawing the rectangle
                self.drawing = False
                # Check if we have valid start and end points that are different
                if self.rect_start and self.rect_end and self.rect_start != self.rect_end:
                    # Create a QRect from the start and end points
                    rect = self._make_rect(self.rect_start, self.rect_end)
                    # Save current state to undo stack before adding new region
                    self.undo_stack.append(list(self.regions))
                    # Clear redo stack since we're adding a new action
                    self.redo_stack.clear()
                    # Add the new rectangle region to the regions list
                    self.regions.append({"type": "rect", "rect": rect})
                    # Update the region polygons for inference calculations
                    self.update_region_polygons()
                # Reset rectangle drawing state
                self.rect_start = None
                self.rect_end = None
                self.current_rect = None
            # Update the video display to reflect changes
            self.video_display.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for updating region shape."""
        if self.drawing and self.rect_start:
            # Use raw mouse coordinates from the video display widget
            pos = event.pos()
            self.rect_end = pos
            self.current_rect = self._make_rect(self.rect_start, self.rect_end)
            self.video_display.update()
        elif self.drawing_polygon and self.current_polygon_points:
            # Update the last point of the polygon in progress
            pos = event.pos()
            self.current_polygon_points[-1] = pos
            self.video_display.update()
        
    def keyReleaseEvent(self, event):
        """Finish polygon drawing when Ctrl is released."""
        if event.key() == Qt.Key_Control and self.drawing_polygon:
            if len(self.current_polygon_points) > 2:
                self.regions.append({
                    "type": "polygon",
                    "points": list(self.current_polygon_points)
                })
                self.update_region_polygons()
            self.drawing_polygon = False
            self.current_polygon_points = []
            self.video_display.update()

    def _make_rect(self, p1, p2):
        """Return a QRect from two points."""
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        return QRect(left, top, right - left, bottom - top)
        
    def _get_video_offset(self):
        """Calculate the offset and scale for centering the video in the widget."""
        if self.current_frame is not None:
            frame_h, frame_w = self.current_frame.shape[:2]
            widget_width = self.video_display.width()
            widget_height = self.video_display.height()
            
            # Calculate scale to fit video while maintaining aspect ratio
            scale = min(widget_width / frame_w, widget_height / frame_h)
            scaled_w = int(frame_w * scale)
            scaled_h = int(frame_h * scale)
            
            # Calculate offset to center the scaled video
            offset_x = (widget_width - scaled_w) // 2
            offset_y = (widget_height - scaled_h) // 2
            
            return offset_x, offset_y
        return 0, 0

    def _setup_video_output(self, video_path, output_dir):
        """Setup video output with timestamp in filename."""
        try:
            video_info = sv.VideoInfo.from_video_path(video_path=video_path)
            
            # Create timestamped filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{base_name}_{timestamp}.mp4"
            self.output_path = os.path.join(output_dir, output_filename)
            
            # Initialize VideoSink
            self.video_sink = sv.VideoSink(target_path=self.output_path, video_info=video_info)
            self.video_sink.__enter__()
            self.should_write_video = True
            
        except Exception as e:
            print(f"Failed to setup video output: {e}")
            self.should_write_video = False
            self.video_sink = None

    def _prepare_new_video_output(self):
        """Prepare a new video output with fresh timestamp if output directory exists."""
        if self.output_dir and os.path.exists(self.output_dir) and self.video_path:
            self._setup_video_output(self.video_path, self.output_dir)

    def _write_frame_to_sink(self, frame):
        """Write a frame to the video sink if enabled."""
        if not self.should_write_video or self.video_sink is None:
            return
        
        try:
            # Ensure frame matches expected output size
            expected_shape = (self.video_sink.video_info.height, self.video_sink.video_info.width)
            if frame.shape[:2] != expected_shape:
                frame = cv2.resize(frame, (expected_shape[1], expected_shape[0]))
            self.video_sink.write_frame(frame)
        except Exception as e:
            print(f"Error writing frame to video sink: {e}")
            
    def _cleanup_video_sink(self):
        """Properly cleanup the video sink and prepare for potential new recording."""
        if self.video_sink is not None:
            try:
                self.video_sink.__exit__(None, None, None)
                print(f"Video saved to: {self.output_path}")
            except Exception as e:
                print(f"Error closing video sink: {e}")
            finally:
                self.video_sink = None
                self.should_write_video = False
                self.output_path = None
    
    def _handle_video_end(self):
        """Handle when video reaches the end."""
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        
        # Finalize current video output
        self._cleanup_video_sink()
        print("Video recording ended - ready for new recording on next play")
        
    def load_video(self, video_path, output_dir=None):
        """Load a video file and prepare for playback and region drawing."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Clean up existing video capture
            if self.cap:
                self.cap.release()
                
            # Clean up existing video sink
            self._cleanup_video_sink()
            
            # Load new video
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self.parent, 
                                     "Error", 
                                     f"Failed to open video file: {video_path}")
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.seek_slider.setMaximum(self.total_frames - 1)
            self.current_frame_number = 0
            self.is_playing = False
            self.video_path = video_path
            self.output_dir = output_dir
            
            # Clear regions and region polygons when loading a new video
            self.regions.clear()
            self.region_polygons.clear()
            self.update()
            
            # Do NOT setup output video here; only do so when recording is started
            self.should_write_video = False
            self.update_record_buttons()
            
            # Load first frame
            self.seek(0)
            self.update()
            self.update_frame_label()
            self.enable_video_region()
            
            # Reset for new video
            self.inference_engine.reset_tracker()
            
        except Exception as e:
            QMessageBox.critical(self.parent, 
                                 "Error", 
                                 f"Failed to load video: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def browse_output(self):
        """Open directory dialog to select output directory."""
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_edit.setText(dir_name)
            self.output_dir = dir_name
            # If video already loaded, update output dir for widget
            if self.video_path:
                self.video_region_widget.load_video(self.video_path, dir_name)
            else:
                self.update_record_buttons()
        else:
            self.update_record_buttons()

    def process_frame_for_inference(self, frame):
        """Process frame for inference if enabled."""
        if not self.inference_enabled or not self.inference_engine:
            return frame
        
        try:
            # Run inference on the current frame
            detections = self.inference_engine.infer(frame)
            # Count objects in defined regions
            detections, region_counts = self.inference_engine.count_objects_in_regions(detections, self.region_polygons)
            # Draw detections on the frame
            frame = self.draw_inference_results(frame, detections, region_counts)
            
        except Exception as e:
            print(f"Inference processing failed: {e}")
            
        return frame
        
    def next_frame(self):
        """Advance to the next frame in the video and update the display."""
        if not self.cap or not self.is_playing:
            return
            
        # Use playback speed
        step = max(1, int(self.playback_speed))
        
        # Read frames according to playback speed
        frame = None
        for _ in range(step):
            ret, frame = self.cap.read()
            if not ret:
                self._handle_video_end()
                return
                
        if frame is not None:
            processed_frame = self.process_frame_for_inference(frame.copy())
            # Only write if recording
            if self.should_write_video:
                self._write_frame_to_sink(processed_frame)
            
            # Update display
            self.current_frame = processed_frame
            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.update()
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_number)
            self.seek_slider.blockSignals(False)
            self.update_frame_label()

    def draw_inference_results(self, frame, detections, region_counts):
        """Draw inference results on the video frame using supervision's BoxAnnotator."""
        if not detections or len(detections) == 0:
            return frame
        try:
            # Get the class names from detections
            class_names = detections.data.get('class_name', [])
            # Get confidences for each detection
            confidences = detections.confidence
            # Try to get tracker IDs if present
            tracker_ids = detections.tracker_id

            if tracker_ids is not None:
                labels = [
                    f"#{int(tid)} {name}: {conf:.2f}" if tid is not None else f"{name}: {conf:.2f}"
                    for name, conf, tid in zip(class_names, confidences, tracker_ids)
                ]
            else:
                labels = [f"{name}: {conf:.2f}" for name, conf in zip(class_names, confidences)]

            # Get selected annotators 
            selected_annotators = self.parent.get_selected_annotators()
            annotators = []
            for key in selected_annotators:
                if key == "BoxAnnotator":
                    annotators.append(sv.BoxAnnotator())
                elif key == "RoundBoxAnnotator":
                    annotators.append(sv.RoundBoxAnnotator())
                elif key == "BoxCornerAnnotator":
                    annotators.append(sv.BoxCornerAnnotator())
                elif key == "ColorAnnotator":
                    annotators.append(sv.ColorAnnotator())
                elif key == "CircleAnnotator":
                    annotators.append(sv.CircleAnnotator())
                elif key == "DotAnnotator":
                    annotators.append(sv.DotAnnotator())
                elif key == "TriangleAnnotator":
                    annotators.append(sv.TriangleAnnotator())
                elif key == "EllipseAnnotator":
                    annotators.append(sv.EllipseAnnotator())
                elif key == "HaloAnnotator":
                    annotators.append(sv.HaloAnnotator())
                elif key == "PercentageBarAnnotator":
                    annotators.append(sv.PercentageBarAnnotator())
                elif key == "MaskAnnotator":
                    annotators.append(sv.MaskAnnotator())
                elif key == "PolygonAnnotator":
                    annotators.append(sv.PolygonAnnotator())
                elif key == "BlurAnnotator":
                    annotators.append(sv.BlurAnnotator())
                elif key == "PixelateAnnotator":
                    annotators.append(sv.PixelateAnnotator())
                elif key == "LabelAnnotator":
                    annotators.append(sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER))
                    
            for annotator in annotators:
                # Only pass labels to LabelAnnotator
                if isinstance(annotator, sv.LabelAnnotator):
                    frame = annotator.annotate(scene=frame, detections=detections, labels=labels)
                else:
                    frame = annotator.annotate(scene=frame, detections=detections)
        except Exception as e:
            print(f"Supervision annotate failed: {e}")
            
        # Draw region polygons
        for idx, poly in enumerate(self.region_polygons):
            if idx < len(region_counts):
                # Get the polygon points
                pts = np.array(list(poly.exterior.coords), np.int32)
                
                # Find the top-left corner of the polygon for text placement
                min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
                
                # Draw the polygon on the frame
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                
                # Calculate y position for text, ensuring it stays within the frame
                text_y = int(min_y) + 90  # Lowered further on the y-axis
                if text_y > frame.shape[0] - 10:
                    text_y = frame.shape[0] - 10
                    
                # Place text at the adjusted position
                cv2.putText(
                    img=frame,
                    text=str(region_counts[idx]),
                    org=(int(min_x), text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 0, 0),
                    thickness=3
                )
                
        return frame

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()
            

class InferenceEngine:
    """Handles model loading, inference, and class filtering."""
    def __init__(self, parent=None):
        self.parent = parent
        
        # Set the device (video_region.base.main_window)
        self.device = "cpu"
        
        # Initialize model and task
        self.model = None
        self.task = None
        
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack()
        
        # Default inference parameters
        self.conf = 0.3
        self.iou = 0.2
        self.area_min = 0.0
        self.area_max = 0.4
        
        self.count_criteria = "Centroid"  # Criteria for counting objects in regions
        self.display_outside = True
        
        self.class_names = []
        self.selected_classes = []

    def load_model(self, model_path, task):
        """Load the YOLO model for inference."""
        # Make cursor busy while loading
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            # Set the task
            self.task = task
            # Load the model using YOLO from ultralytics
            self.model = YOLO(model_path, task=self.task)
            # Store class names from the model
            self.class_names = list(self.model.names.values())
            
            # Run a dummy inference to ensure the model is loaded correctly
            self.model(np.zeros((640, 640, 3), dtype=np.uint8))
            
            # Set the device for inference
            self.set_device()
            
            QMessageBox.information(self.parent,
                                    "Model Loaded",
                                    "Model loaded successfully.")
                        
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self.parent, 
                                 "Model Load Error",
                                 "Failed to load model (see console for details)")
            
        finally:
            # Make cursor normal
            QApplication.restoreOverrideCursor()
            
    def set_device(self):
        """Set the device for inference."""
        self.device = self.parent.parent.main_window.device

    def set_selected_classes(self, class_indices):
        """Set the selected classes for inference."""
        self.selected_classes = class_indices
        
    def set_inference_params(self, conf, iou, area_min, area_max):
        """Set inference parameters for the video region."""
        self.conf = conf
        self.iou = iou
        self.area_min = area_min
        self.area_max = area_max

    def set_count_criteria(self, count_criteria):
        """Set the criteria for counting objects in regions."""
        self.count_criteria = count_criteria
        
    def set_display_outside_detections(self, display_outside):
        """Set whether to display detections outside the regions."""
        self.display_outside = display_outside

    def infer(self, frame):
        """Run inference on a single frame with the current model."""
        if self.model is None:
            return None
        
        # Detect, and filter results based on confidence and IoU
        results = self.model(frame, 
                             conf=self.conf, 
                             iou=self.iou, 
                             classes=self.selected_classes,
                             half=True,
                             device=self.device)[0]
        
        # Convert results to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker with detections
        detections = self.update_tracker(detections)
        
        # Apply area filter to detections
        detections = self.apply_area_filter(frame, detections)
            
        return detections
    
    def update_tracker(self, detections):
        """Update the tracker with the current detections."""
        tracker_detections = self.tracker.update_with_detections(detections)
        if tracker_detections:
            return tracker_detections
        return detections
    
    def apply_area_filter(self, frame, detections):
        """Filter detections based on area thresholds."""
        if detections is None or len(detections) == 0:
            return detections
        
        # Calculate the area of the frame
        height, width = frame.shape[:2]
        frame_area = height * width
        
        # Filter detections based on relative area
        detections = detections[(detections.area / frame_area) <= self.area_max]
        detections = detections[(detections.area / frame_area) >= self.area_min]
        
        return detections
            
    def count_objects_in_regions(self, detections, region_polygons):
        """Count objects in each region based on supervision Detections."""
        if not region_polygons or detections is None or len(detections) == 0:
            region_counts = [0 for _ in region_polygons]
            return detections, region_counts

        # Get the number of regions
        region_counts = [0 for _ in region_polygons]
        
        # Use detection boxes for region assignment
        boxes = detections.xyxy  # shape (n, 4)
        
        # Create mask to track which detections are inside regions
        inside_mask = np.zeros(len(boxes), dtype=bool)
        
        # Loop through each detection box
        for i, box in enumerate(boxes):
            
            # Check if just the centroid is in inside the region
            if self.count_criteria == "Centroid":
                # Check if centroid is inside the region
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                for idx, poly in enumerate(region_polygons):
                    if poly.contains(Point(center)):
                        region_counts[idx] += 1
                        inside_mask[i] = True
                        break  # Count in first matching region only
                    
            # Check if entire bounding box is inside the region
            elif self.count_criteria == "Bounding Box":
                box_corners = [
                    (box[0], box[1]),  # top-left
                    (box[2], box[1]),  # top-right
                    (box[2], box[3]),  # bottom-right
                    (box[0], box[3])   # bottom-left
                ]
                for idx, poly in enumerate(region_polygons):
                    if all(poly.contains(Point(corner)) for corner in box_corners):
                        region_counts[idx] += 1
                        inside_mask[i] = True
                        break  # Count in first matching region only
        
        # Filter detections based on display_outside setting
        if not self.display_outside:
            detections = detections[inside_mask]
    
        return detections, region_counts
    
    def reset_tracker(self):
        """Reset the ByteTrack tracker."""
        if self.tracker:
            self.tracker.reset()
            print("Tracker reset")
        else:
            print("No tracker initialized")

            
class Base(QDialog):
    """Dialog for video inference with region selection and parameter controls."""
    def __init__(self, main_window, parent=None):
        """Initialize the Video Inference dialog."""
        super().__init__(parent)
        self.main_window = main_window
        
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Video Inference")
        
        # Optionally set a minimum size
        self.setMinimumSize(800, 600)
        
        # Initialize device as 'cpu' by default
        self.device = 'cpu'
                
        # Initialize parameters
        self.video_path = ""
        self.output_dir = ""
        
        self.model_path = ""
        
        self.task = None                            # Task parameter, modified in subclasses
        
        self.uncertainty_thresh = 0.30
        self.iou_thresh = 0.20
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
                 
        self.video_region_widget = None             # Initialized in setup_video_layout

        # Main layout
        self.layout = QHBoxLayout(self)
        self.controls_layout = QVBoxLayout()
        self.layout.addLayout(self.controls_layout, 30)
        self.video_layout = QVBoxLayout()
        self.layout.addLayout(self.video_layout, 70)

        # Setup the input layout
        self.setup_input_layout()
        # Setup the output layout
        self.setup_output_layout()
        # Setup the model and parameters layout
        self.setup_model_layout()
        # Setup the video player widget
        self.setup_video_layout()
        # Setup regions control layout
        self.setup_regions_layout()
        # Setup annotators control layout
        self.setup_annotators_layout()
        # Setup inference controls layout
        self.setup_inference_layout()
        # Setup Run/Cancel buttons
        self.setup_buttons_layout()

    def setup_input_layout(self):
        """Setup the input video group with a file browser using QFormLayout."""
        group_box = QGroupBox("Input Parameters")
        layout = QFormLayout()

        self.input_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(browse_btn)
        input_widget = QWidget()
        input_widget.setLayout(input_layout)

        layout.addRow(QLabel("Input Video:"), input_widget)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_output_layout(self):
        """Setup the output directory group with a file browser using QFormLayout."""
        group_box = QGroupBox("Output Parameters")
        layout = QFormLayout()

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Provide directory to write inferenced video...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse_btn)
        output_widget = QWidget()
        output_widget.setLayout(output_layout)

        layout.addRow(QLabel("Output Directory:"), output_widget)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_model_layout(self):
        """Setup the model input, parameters, and class filter in a single group using QFormLayout."""
        group_box = QGroupBox("Model and Parameters")
        form_layout = QFormLayout()

        # Model path input
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(browse_btn)
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        form_layout.addRow(QLabel("Model Path:"), model_widget)

        # Searchable Class Filter
        class_filter_layout = QVBoxLayout()
        self.class_filter_search = QLineEdit()
        self.class_filter_search.setPlaceholderText("Search classes...")
        self.class_filter_search.textChanged.connect(self.filter_class_list)
        class_filter_layout.addWidget(self.class_filter_search)
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        class_filter_layout.addWidget(self.class_filter_widget)
        class_filter_widget_container = QWidget()
        class_filter_widget_container.setLayout(class_filter_layout)
        form_layout.addRow(QLabel("Class Filter:"), class_filter_widget_container)

        # Select All / Deselect All buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        form_layout.addRow(QLabel(""), btn_widget)

        # Parameter sliders (IoU, uncertainty, area)
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_thresh_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        uncertainty_layout = QHBoxLayout()
        uncertainty_layout.addWidget(self.uncertainty_thresh_slider)
        uncertainty_layout.addWidget(self.uncertainty_thresh_label)
        uncertainty_widget = QWidget()
        uncertainty_widget.setLayout(uncertainty_layout)
        form_layout.addRow(QLabel("Uncertainty Threshold:"), uncertainty_widget)

        self.iou_thresh_slider = QSlider(Qt.Horizontal)
        self.iou_thresh_slider.setRange(0, 100)
        self.iou_thresh_slider.setValue(int(self.iou_thresh * 100))
        self.iou_thresh_slider.valueChanged.connect(self.update_iou_label)
        self.iou_thresh_label = QLabel(f"{self.iou_thresh:.2f}")
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(self.iou_thresh_slider)
        iou_layout.addWidget(self.iou_thresh_label)
        iou_widget = QWidget()
        iou_widget.setLayout(iou_layout)
        form_layout.addRow(QLabel("IoU Threshold:"), iou_widget)

        self.area_threshold_min_slider = QSlider(Qt.Horizontal)
        self.area_threshold_min_slider.setRange(0, 100)
        self.area_threshold_min_slider.setValue(int(self.area_thresh_min * 100))
        self.area_threshold_min_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_min_label = QLabel(f"{self.area_thresh_min:.2f}")
        area_min_layout = QHBoxLayout()
        area_min_layout.addWidget(self.area_threshold_min_slider)
        area_min_layout.addWidget(self.area_threshold_min_label)
        area_min_widget = QWidget()
        area_min_widget.setLayout(area_min_layout)
        form_layout.addRow(QLabel("Area Threshold Min:"), area_min_widget)

        self.area_threshold_max_slider = QSlider(Qt.Horizontal)
        self.area_threshold_max_slider.setRange(0, 100)
        self.area_threshold_max_slider.setValue(int(self.area_thresh_max * 100))
        self.area_threshold_max_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_max_label = QLabel(f"{self.area_thresh_max:.2f}")
        area_max_layout = QHBoxLayout()
        area_max_layout.addWidget(self.area_threshold_max_slider)
        area_max_layout.addWidget(self.area_threshold_max_label)
        area_max_widget = QWidget()
        area_max_widget.setLayout(area_max_layout)
        form_layout.addRow(QLabel("Area Threshold Max:"), area_max_widget)

        group_box.setLayout(form_layout)
        self.controls_layout.addWidget(group_box)

    def filter_class_list(self, text):
        """Filter the class filter QListWidget based on the search text."""
        text = text.lower()
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setHidden(text not in item.text().lower())

    def setup_video_layout(self):
        """Setup the video region widget directly without an external group box."""
        self.video_region_widget = VideoRegionWidget(self)
        self.video_layout.addWidget(self.video_region_widget)

    def setup_regions_layout(self):
        """Setup the regions control group."""
        group_box = QGroupBox("Regions")
        layout = QHBoxLayout()

        clear_btn = QPushButton("Clear Regions")
        clear_btn.setFocusPolicy(Qt.NoFocus)
        clear_btn.clicked.connect(self.clear_regions)
        layout.addWidget(clear_btn)
        
        # Add stretch to push everything to the left
        layout.addStretch()
        
        # Count criteria dropdown with label
        layout.addWidget(QLabel("Count Criteria:"))
        self.count_criteria_combo = QComboBox()
        self.count_criteria_combo.addItems(["Centroid", "Bounding Box"])
        self.count_criteria_combo.setCurrentIndex(0)
        self.count_criteria_combo.currentIndexChanged.connect(self.update_region_parameters)
        layout.addWidget(self.count_criteria_combo)

        # Dropdown for display detections outside regions with label
        layout.addWidget(QLabel("Show Outside Detections:"))
        self.display_outside_combo = QComboBox()
        self.display_outside_combo.addItems(["True", "False"])
        self.display_outside_combo.setCurrentIndex(0)
        self.display_outside_combo.currentIndexChanged.connect(self.update_region_parameters)
        layout.addWidget(self.display_outside_combo)
        
        # Add some spacing
        layout.addSpacing(10)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_annotators_layout(self):
        """Setup the annotator selection layout using a QListWidget with checkable items."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def setup_inference_layout(self):
        """Setup the inference control group with Enable/Disable buttons."""
        group_box = QGroupBox("Inference Controls")
        layout = QHBoxLayout()
        
        self.enable_inference_btn = QPushButton("Enable Inference")
        self.enable_inference_btn.clicked.connect(self.enable_inference)
        self.enable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        layout.addWidget(self.enable_inference_btn)
        
        self.disable_inference_btn = QPushButton("Disable Inference")
        self.disable_inference_btn.clicked.connect(self.disable_inference)
        self.disable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        self.disable_inference_btn.setEnabled(False)           # Initially disabled
        layout.addWidget(self.disable_inference_btn)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Setup the Exit button at the bottom of the controls layout."""
        btn_layout = QHBoxLayout()
        
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.reject)
        self.exit_btn.setFocusPolicy(Qt.NoFocus)                # Prevent focus/highlighting
        btn_layout.addWidget(self.exit_btn)
        
        self.controls_layout.addLayout(btn_layout)
        
    def browse_video(self):
        """Open file dialog to select input video (filtered to common formats)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        if file_name:
            self.input_edit.setText(file_name)
            self.video_path = file_name
            self.video_region_widget.load_video(file_name, self.output_dir)

    def browse_output(self):
        """Open directory dialog to select output directory."""
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_edit.setText(dir_name)
            self.output_dir = dir_name
            # If video already loaded, update output dir for widget
            if self.video_path:
                self.video_region_widget.load_video(self.video_path, dir_name)
            else:
                self.update_record_buttons()
        else:
            self.update_record_buttons()

    def browse_model(self):
        """Open file dialog to select model file (filtered to .pt, .pth)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        if file_name:
            self.model_edit.setText(file_name)
            self.model_path = file_name
            self.video_region_widget.inference_engine.load_model(file_name, task=self.task)
            self.populate_class_filter()
        
    def get_selected_annotators(self):
        """Return a list of selected annotator class names from the QListWidget."""
        selected = []
        for i in range(self.annotator_list_widget.count()):
            item = self.annotator_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def initialize_uncertainty_threshold(self):
        """Initialize the confidence threshold slider with the current value"""
        current_value = self.main_window.get_uncertainty_thresh()
        self.uncertainty_thresh_slider.setValue(int(current_value * 100))
        self.uncertainty_thresh = current_value

    def initialize_iou_threshold(self):
        """Initialize the IOU threshold slider with the current value"""
        current_value = self.main_window.get_iou_thresh()
        self.iou_thresh_slider.setValue(int(current_value * 100))
        self.iou_thresh = current_value

    def initialize_area_threshold(self):
        """Initialize the area threshold range slider"""
        current_min, current_max = self.main_window.get_area_thresh()
        self.area_threshold_min_slider.setValue(int(current_min * 100))
        self.area_threshold_max_slider.setValue(int(current_max * 100))
        self.area_thresh_min = current_min
        self.area_thresh_max = current_max

    def update_uncertainty_label(self, value):
        """Update confidence threshold and label"""
        value = value / 100.0
        self.uncertainty_thresh = value
        self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_thresh_label.setText(f"{value:.2f}")
        self.update_inference_parameters()

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        self.main_window.update_iou_thresh(value)
        self.iou_thresh_label.setText(f"{value:.2f}")
        self.update_inference_parameters()

    def update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val = self.area_threshold_min_slider.value()
        max_val = self.area_threshold_max_slider.value()
        if min_val > max_val:
            min_val = max_val
            self.area_threshold_min_slider.setValue(min_val)
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.main_window.update_area_thresh(self.area_thresh_min, self.area_thresh_max)
        self.area_threshold_min_label.setText(f"{self.area_thresh_min:.2f}")
        self.area_threshold_max_label.setText(f"{self.area_thresh_max:.2f}")
        self.update_inference_parameters()

    def update_inference_parameters(self):
        """Update inference parameters in the video region widget."""
        self.video_region_widget.inference_engine.set_inference_params(
            self.uncertainty_thresh,
            self.iou_thresh,
            self.area_thresh_min,
            self.area_thresh_max
        )
        
    def update_region_parameters(self):
        """Update region parameters based on the selected count criteria and display outside detections."""
        count_criteria = self.count_criteria_combo.currentText()
        display_outside = self.display_outside_combo.currentText() == "True"
        # Update the inference engine with the selected criteria
        self.video_region_widget.inference_engine.set_count_criteria(count_criteria)
        self.video_region_widget.inference_engine.set_display_outside_detections(display_outside)

    def populate_class_filter(self):
        """Populate the class filter widget with class names from the loaded model."""
        self.class_filter_widget.clear()
        for idx, name in enumerate(self.video_region_widget.inference_engine.class_names):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, idx)
            item.setCheckState(Qt.Checked)  # Select all by default
            self.class_filter_widget.addItem(item)
        self.class_filter_widget.itemChanged.connect(self.update_selected_classes)
        self.update_selected_classes()  # Ensure selected_classes is up to date

    def update_selected_classes(self):
        """Update the selected classes based on the class filter widget."""
        selected = []
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.video_region_widget.inference_engine.set_selected_classes(selected)

    def select_all_classes(self):
        """Select all classes in the class filter widget."""
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setCheckState(Qt.Checked)
        self.update_selected_classes()

    def deselect_all_classes(self):
        """Deselect all classes in the class filter widget."""
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            item.setCheckState(Qt.Unchecked)
        self.update_selected_classes()
        
    def clear_regions(self):
        """Clear all regions from the video region widget and update display."""
        self.video_region_widget.clear_regions()

    def enable_inference(self):
        """Enable inference on the video region."""
        if not self.video_region_widget.inference_engine.model:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a model before enabling inference.")
            return
        
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(True)
        self.enable_inference_btn.setEnabled(False)
        self.disable_inference_btn.setEnabled(True)
        
    def disable_inference(self):
        """Disable inference on the video region."""
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(False)
        self.enable_inference_btn.setEnabled(True)
        self.disable_inference_btn.setEnabled(False)
        
        # Refresh the current frame without inference
        self.video_region_widget.seek(self.video_region_widget.current_frame_number)
