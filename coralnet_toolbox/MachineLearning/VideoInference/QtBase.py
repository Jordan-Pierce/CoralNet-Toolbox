import cv2
import numpy as np

from shapely.geometry import Polygon, Point

from ultralytics import YOLO

from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, 
                             QWidget, QGridLayout, QListWidget, QListWidgetItem, 
                             QAbstractItemView, QFormLayout, QTabWidget, 
                             QComboBox, QCheckBox, QSpacerItem, QSizePolicy)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RegionManager:
    """Manages region creation, editing, and storage."""
    def __init__(self):
        self.regions = []  # List of polygons (list of QPoint)
        # Add more region management logic as needed

    def add_region(self, polygon):
        self.regions.append(polygon)

    def clear_regions(self):
        self.regions.clear()

    # Add methods for editing, deleting, moving regions, etc.


class VideoRegionWidget(QWidget):
    """Widget for displaying video, playback controls, and drawing/editing rectangular regions only."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.frame = None
        self.pixmap = None
        self.regions = []  # List of QRect (or 4-point polygons)
        self.drawing = False
        self.rect_start = None  # QPoint
        self.rect_end = None    # QPoint
        self.current_rect = None  # QRect
        self.selected_region = None
        self.setMouseTracking(True)
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

        # Layout: video area (fills widget), controls at bottom
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addStretch(1)  # Video area will be drawn in paintEvent
        controls = QHBoxLayout()
        
        # Step Backward Button
        self.step_back_btn = QPushButton("⏮")
        self.step_back_btn.setToolTip("Step Backward")
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        controls.addWidget(self.step_back_btn)

        # Play button with icon 
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPlay))
        self.play_btn.setToolTip("Play")
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        controls.addWidget(self.play_btn)
        
        # Pause button with icon
        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(self.style().standardIcon(self.style().SP_MediaPause))
        self.pause_btn.setToolTip("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        controls.addWidget(self.pause_btn)

        # Step Forward Button 
        self.step_fwd_btn = QPushButton("⏭")
        self.step_fwd_btn.setToolTip("Step Forward")
        self.step_fwd_btn.clicked.connect(self.step_forward)
        self.step_fwd_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        controls.addWidget(self.step_fwd_btn)

        # Stop (reset to first frame) button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(self.style().standardIcon(self.style().SP_MediaStop))
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.clicked.connect(self.reset_to_first_frame)
        self.stop_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        controls.addWidget(self.stop_btn)

        # Slider for seeking through video
        controls.addSpacing(8)
        controls.addStretch(1)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek)
        self.seek_slider.setMinimumWidth(600)
        self.seek_slider.setMaximumWidth(1000)
        controls.addWidget(self.seek_slider)
        controls.addStretch(1)
        controls.addSpacing(8)

        # Set maximum size for buttons
        max_btn_size = 32
        for btn in [self.step_back_btn, self.play_btn, self.pause_btn, self.step_fwd_btn, self.stop_btn]:
            btn.setMaximumSize(max_btn_size, max_btn_size)

        # Frame Label
        self.frame_label = QLabel("Frame: 0 / 0")
        controls.addWidget(self.frame_label)

        # Playback Speed Dropdown
        self.speed_dropdown = QComboBox()
        self.speed_dropdown.addItems(["0.5x", "1x", "2x"])
        self.speed_dropdown.setCurrentIndex(1)
        self.speed_dropdown.currentIndexChanged.connect(self.change_speed)
        self.speed_dropdown.setMaximumWidth(80)
        controls.addWidget(self.speed_dropdown)

        self.layout.addLayout(controls)

        # At the end of __init__, disable controls by default
        self.disable_video_region()

    def enable_video_region(self):
        """Enable all controls in the video region widget."""
        self.setEnabled(True)
        self.step_back_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.step_fwd_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.speed_dropdown.setEnabled(True)
        self.frame_label.setEnabled(True)

    def disable_video_region(self):
        """Disable all controls in the video region widget."""
        self.setEnabled(False)
        self.step_back_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.step_fwd_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.speed_dropdown.setEnabled(False)
        self.frame_label.setEnabled(False)

    def load_video(self, video_path):
        """Load a video file and prepare for playback and region drawing."""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.seek_slider.setMaximum(self.total_frames - 1)
        self.current_frame_number = 0
        self.is_playing = False
        self.seek(0)
        self.update()
        self.update_frame_label()

        # Enable controls when video is loaded
        self.enable_video_region()

    def next_frame(self):
        """Advance to the next frame in the video and update the display."""
        if self.cap and self.is_playing:
            #  Use playback speed
            step = int(self.playback_speed)
            for _ in range(step):
                ret, frame = self.cap.read()
                if not ret:
                    self.timer.stop()
                    self.is_playing = False
                    return
            if ret:
                self.current_frame = frame
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.update()
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(self.current_frame_number)
                self.seek_slider.blockSignals(False)
                self.update_frame_label()
            else:
                self.timer.stop()
                self.is_playing = False

    def play_video(self):
        """Play the video from the current position."""
        if not self.is_playing:
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

    def seek(self, frame_number):
        """Seek to a specific frame in the video."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_number = frame_number
                self.update()
                self.update_frame_label()

    def display_frame(self, frame):
        """Display a processed frame in the video widget."""
        self.current_frame = frame
        self.update()
        self.update_frame_label()

    def update_pixmap(self):
        pass  # No longer needed

    def paintEvent(self, event):
        """Draw the video frame centered, and rectangular regions."""
        painter = QPainter(self)
        if self.current_frame is not None:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            widget_width = self.width()
            widget_height = self.height() - 50  # Reserve space for controls
            scaled = pixmap.scaled(widget_width, widget_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (widget_width - scaled.width()) // 2
            y = (widget_height - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0
        # Draw rectangles
        if self.show_regions:
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            for rect in self.regions:
                r = rect.translated(offset_x, offset_y)
                painter.drawRect(r)
        # Draw current rectangle
        if self.drawing and self.current_rect:
            pen = QPen(Qt.green, 2, Qt.DashLine)
            painter.setPen(pen)
            r = self.current_rect.translated(offset_x, offset_y)
            painter.drawRect(r)

    def mousePressEvent(self, event):
        """Handle mouse press events for drawing regions."""
        # Adjust for centering offset
        offset_x, offset_y = self._get_video_offset()
        pos = event.pos() - QPoint(offset_x, offset_y)
        if event.button() == Qt.LeftButton:
            if not self.drawing:
                self.drawing = True
                self.rect_start = pos
                self.rect_end = pos
                self.current_rect = None
            else:
                self.drawing = False
                if self.rect_start and self.rect_end and self.rect_start != self.rect_end:
                    rect = self._make_rect(self.rect_start, self.rect_end)
                    # Push to undo stack
                    self.undo_stack.append(list(self.regions))
                    self.redo_stack.clear()
                    self.regions.append(rect)
                self.rect_start = None
                self.rect_end = None
                self.current_rect = None
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events for updating region shape."""
        if self.drawing and self.rect_start:
            offset_x, offset_y = self._get_video_offset()
            pos = event.pos() - QPoint(offset_x, offset_y)
            self.rect_end = pos
            self.current_rect = self._make_rect(self.rect_start, self.rect_end)
            self.update()

    def _get_video_offset(self):
        """Calculate the offset for centering the video in the widget."""
        if self.current_frame is not None:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            widget_width = self.width()
            widget_height = self.height() - 50
            pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
            scaled = pixmap.scaled(widget_width, widget_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (widget_width - scaled.width()) // 2
            y = (widget_height - scaled.height()) // 2
            return x, y
        return 0, 0

    def mouseDoubleClickEvent(self, event):
        """Ignore double click for rectangle creation."""
        pass

    def _make_rect(self, p1, p2):
        """Return a QRect from two points."""
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        from PyQt5.QtCore import QRect
        return QRect(left, top, right - left, bottom - top)

    # Step Forward/Backward
    def step_forward(self):
        if self.cap:
            next_frame = min(self.current_frame_number + 1, self.total_frames - 1)
            self.seek(next_frame)

    def step_backward(self):
        if self.cap:
            prev_frame = max(self.current_frame_number - 1, 0)
            self.seek(prev_frame)

    # Playback Speed
    def change_speed(self, idx):
        speeds = [0.5, 1.0, 2.0]
        self.playback_speed = speeds[idx]
        if self.is_playing:
            self.toggle_pause()
            self.toggle_play()

    # Region Visibility
    def set_region_visibility(self, visible: bool):
        self.show_regions = visible
        self.update()

    # Reset to First Frame
    def reset_to_first_frame(self):
        if self.cap:
            self.seek(0)
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.seek_slider.setValue(0)  # Update slider to first frame
            self.update_frame_label()     # Update frame count label

    # Frame Label
    def update_frame_label(self):
        self.frame_label.setText(f"Frame: {self.current_frame_number + 1} / {self.total_frames}")

    # Undo/Redo for regions
    def undo_region(self):
        if self.undo_stack:
            self.redo_stack.append(list(self.regions))
            self.regions = self.undo_stack.pop()
            self.update()

    def redo_region(self):
        if self.redo_stack:
            self.undo_stack.append(list(self.regions))
            self.regions = self.redo_stack.pop()
            self.update()


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
        
        # Initialize parameters
        self.video_path = ""
        self.output_dir = ""
        self.model_path = ""
        
        self.task = None  # Task parameter, default is None
        
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        
        # Track inference state
        self.inference_state = False  # False = disabled, True = enabled
        
        self.region_manager = RegionManager()
        self.inference_engine = InferenceEngine()

        # Main layout
        self.layout = QHBoxLayout(self)
        self.controls_layout = QVBoxLayout()
        self.layout.addLayout(self.controls_layout, 30)
        self.video_layout = QVBoxLayout()
        self.layout.addLayout(self.video_layout, 70)

        # Setup the input layout
        self.setup_input_group()
        # Setup the output layout
        self.setup_output_group()
        # Setup the model layout
        self.setup_model_layout()
        # Setup the parameters layout
        self.setup_parameters_layout()
        # Setup the class filter layout
        self.setup_class_layout()
        # Setup the video player widget
        self.setup_video_layout()
        # Setup regions control layout
        self.setup_regions_layout()
        # Setup inference controls layout
        self.setup_inference_layout()
        # Setup Run/Cancel buttons
        self.setup_buttons_layout()
        
        # Initialize thresholds
        self.initialize_thresholds()

        # Add inference thread
        self.inference_thread = InferenceThread()
        self.inference_thread.frame_processed.connect(self.update_frame_with_results)
        self.inference_thread.finished.connect(self.on_inference_finished)

    def setup_input_group(self):
        """Setup the input video group with a file browser."""
        group_box = QGroupBox("Input Parameters")
        layout = QHBoxLayout()
        
        self.input_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        layout.addWidget(QLabel("Input Video:"))
        layout.addWidget(self.input_edit)
        layout.addWidget(browse_btn)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_output_group(self):
        """Setup the output directory group with a file browser."""
        group_box = QGroupBox("Output Parameters")
        layout = QHBoxLayout()
        
        self.output_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        layout.addWidget(QLabel("Output Directory:"))
        layout.addWidget(self.output_edit)
        layout.addWidget(browse_btn)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_model_layout(self):
        """Setup the model input group with a file browser."""
        group_box = QGroupBox("Model Input")
        layout = QGridLayout()
        
        self.model_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        layout.addWidget(QLabel("Model Path:"), 0, 0)
        layout.addWidget(self.model_edit, 0, 1)
        layout.addWidget(browse_btn, 0, 2)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_parameters_layout(self):
        """
        Setup parameter control section in a group box.
        """
        group_box = QGroupBox("Model Parameters")
        layout = QFormLayout()

        # Confidence threshold controls (instead of uncertainty)
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_thresh_slider.setTickInterval(10)
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        self.uncertainty_thresh_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        layout.addRow("Uncertainty Threshold", self.uncertainty_thresh_slider)
        layout.addRow("", self.uncertainty_thresh_label)

        # IoU threshold controls
        self.iou_thresh_slider = QSlider(Qt.Horizontal)
        self.iou_thresh_slider.setRange(0, 100)
        self.iou_thresh_slider.setValue(int(self.iou_thresh * 100))
        self.iou_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_thresh_slider.setTickInterval(10)
        self.iou_thresh_slider.valueChanged.connect(self.update_iou_label)
        self.iou_thresh_label = QLabel(f"{self.iou_thresh:.2f}")
        layout.addRow("IoU Threshold", self.iou_thresh_slider)
        layout.addRow("", self.iou_thresh_label)

        # Area threshold controls
        self.area_threshold_min_slider = QSlider(Qt.Horizontal)
        self.area_threshold_min_slider.setRange(0, 100)
        self.area_threshold_min_slider.setValue(int(self.area_thresh_min * 100))
        self.area_threshold_min_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_min_slider.setTickInterval(10)
        self.area_threshold_min_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_max_slider = QSlider(Qt.Horizontal)
        self.area_threshold_max_slider.setRange(0, 100)
        self.area_threshold_max_slider.setValue(int(self.area_thresh_max * 100))
        self.area_threshold_max_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_max_slider.setTickInterval(10)
        self.area_threshold_max_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        layout.addRow("Area Threshold Min", self.area_threshold_min_slider)
        layout.addRow("Area Threshold Max", self.area_threshold_max_slider)
        layout.addRow("", self.area_threshold_label)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)
        
    def closeEvent(self, event):
        """Ensure inference thread is stopped before closing the dialog."""
        if self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait()
        super().closeEvent(event)

    def setup_class_layout(self):
        """Setup the class filter group box and controls."""
        group_box = QGroupBox("Class Filter")
        layout = QVBoxLayout()

        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.class_filter_widget)

        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_classes)
        self.deselect_all_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        layout.addLayout(btn_layout)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_video_layout(self):
        """Setup the video region widget inside a group box (no tabs)."""
        group_box = QGroupBox("Video Player")
        vbox = QVBoxLayout()
        
        self.video_region_widget = VideoRegionWidget(self)
        vbox.addWidget(self.video_region_widget)
        
        group_box.setLayout(vbox)
        self.video_layout.addWidget(group_box)

        # Connect video widget controls
        self.video_region_widget.play_btn.clicked.connect(self.toggle_inference)
        self.video_region_widget.seek_slider.sliderMoved.connect(self.seek_inference)
        self.video_region_widget.speed_dropdown.currentIndexChanged.connect(self.update_playback_speed)

    def setup_regions_layout(self):
        """Setup the regions control group with a clear button."""
        group_box = QGroupBox("Regions")
        layout = QHBoxLayout()

        #  Show/Hide Regions button (leftmost)
        self.region_vis_btn = QPushButton("Hide Regions")
        self.region_vis_btn.setCheckable(True)
        self.region_vis_btn.setChecked(True)
        
        def toggle_region_btn():
            visible = self.region_vis_btn.isChecked()
            self.video_region_widget.set_region_visibility(visible)
            self.region_vis_btn.setText("Hide Regions" if visible else "Show Regions")
            
        self.region_vis_btn.clicked.connect(toggle_region_btn)
        layout.addWidget(self.region_vis_btn)

        clear_btn = QPushButton("Clear Regions")
        clear_btn.clicked.connect(self.clear_regions)
        layout.addWidget(clear_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.video_region_widget.undo_region)
        layout.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.video_region_widget.redo_region)
        layout.addWidget(self.redo_btn)

        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_inference_layout(self):
        """Setup the inference control group with Enable/Disable buttons."""
        group_box = QGroupBox("Inference Controls")
        layout = QHBoxLayout()
        
        self.enable_inference_btn = QPushButton("Enable Inference")
        self.enable_inference_btn.clicked.connect(self.run_inference_on_video)
        self.enable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        layout.addWidget(self.enable_inference_btn)
        
        self.disable_inference_btn = QPushButton("Disable Inference")
        self.disable_inference_btn.clicked.connect(self.stop_inference)
        self.disable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        self.disable_inference_btn.setEnabled(False)  # Initially disabled
        layout.addWidget(self.disable_inference_btn)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Setup the Exit button at the bottom of the controls layout."""
        btn_layout = QHBoxLayout()
        
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.reject)
        self.exit_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        
        btn_layout.addWidget(self.exit_btn)
        
        self.controls_layout.addLayout(btn_layout)

    def clear_regions(self):
        """Clear all regions from the video region widget and update display."""
        self.video_region_widget.regions.clear()
        self.video_region_widget.update()

    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.initialize_uncertainty_threshold()
        self.initialize_iou_threshold()
        self.initialize_area_threshold()

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

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        self.main_window.update_iou_thresh(value)
        self.iou_thresh_label.setText(f"{value:.2f}")

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
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        
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
            self.video_region_widget.load_video(file_name)

    def browse_output(self):
        """Open directory dialog to select output directory."""
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_edit.setText(dir_name)
            self.output_dir = dir_name

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
            self.inference_engine.load_model(file_name, task=self.task)
            self.populate_class_filter()

    def populate_class_filter(self):
        """Populate the class filter widget with class names from the loaded model."""
        self.class_filter_widget.clear()
        for idx, name in enumerate(self.inference_engine.class_names):
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
        self.inference_engine.set_selected_classes(selected)

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
        
    def update_inference_state(self, state):
        """Update the inference state and adjust UI elements accordingly."""
        self.inference_state = state

    def toggle_inference(self):
        """Toggle inference thread between pause and resume states."""
        if self.inference_thread.isRunning():
            if self.inference_thread._pause:
                self.inference_thread.resume()
                self.video_region_widget.toggle_play_pause()
            else:
                self.inference_thread.pause()
                self.video_region_widget.toggle_play_pause()

    def seek_inference(self, frame_num):
        """Seek inference thread and video widget to a specific frame number."""
        if self.inference_thread.isRunning():
            self.inference_thread.seek(frame_num)
            self.video_region_widget.seek(frame_num)

    def update_playback_speed(self, idx):
        """Update playback speed for video and resume inference if running."""
        speeds = [0.5, 1.0, 2.0]
        self.video_region_widget.playback_speed = speeds[idx]
        if self.inference_thread.isRunning():
            self.inference_thread.resume()

    def update_frame_with_results(self, frame, region_counts, results):
        """Draw region annotations and counts on the frame and display it."""
        # First draw the detection results
        if results and len(results) > 0:
            result = results[0]  # Get the first result
            
            # Draw bounding boxes if they exist
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls[i])
                    confidence = conf[i]
                    
                    # Get class name
                    class_name = self.inference_engine.class_names[class_id] if class_id < len(self.inference_engine.class_names) else str(class_id)
                    
                    # Assign different colors for different classes
                    color_factor = (class_id * 50) % 255
                    color = (color_factor, 255 - color_factor, 150)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw region annotations on the frame
        for idx, poly in enumerate(self.region_polygons):
            pts = np.array(list(poly.exterior.coords), np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            centroid = poly.centroid
            cv2.putText(frame, str(region_counts[idx]),
                        (int(centroid.x), int(centroid.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        # Display the frame
        self.video_region_widget.display_frame(frame)

    def run_inference_on_video(self):
        """Start threaded inference on the loaded video using selected regions and thresholds."""
        if not self.model_path or not self.video_path:
            return
        # Prepare regions
        self.region_polygons = [
            Polygon([
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.right(), rect.bottom()),
                (rect.left(), rect.bottom())
            ]) for rect in self.video_region_widget.regions
        ]
        # Configure thread
        self.inference_thread.setup(
            self.video_path,
            self.inference_engine,
            self.region_polygons,
            {
                'conf': self.uncertainty_thresh,
                'iou': self.iou_thresh
            }
        )
        # Start processing
        self.inference_thread.start()
        self.video_region_widget.play_video()
        
        # Update inference state
        self.update_inference_state(True)

    def stop_inference(self):
        """Stop the inference thread if it is running."""
        if self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait()
            self.video_region_widget.pause_video()
            
            # Update inference state
            self.update_inference_state(False)

    def on_inference_finished(self):
        """Handle actions after inference thread finishes processing video."""
        self.video_region_widget.reset_to_first_frame()
        self.inference_thread.quit()
        
        

class InferenceEngine:
    """Handles model loading, inference, and class filtering."""
    def __init__(self):
        self.model = None
        self.class_names = []
        self.selected_classes = []
        self._thread_safe_infer = None

    def load_model(self, model_path, task):
        """Load the YOLO model for inference."""
        from ultralytics.utils import ThreadingLocked
        self.model = YOLO(model_path, task=task if task else None)
        self.class_names = list(self.model.names.values())
        
        # Decorate the infer method for thread safety
        @ThreadingLocked()
        def thread_safe_infer(frame, conf, iou, selected_classes):
            return self.model.track(frame, persist=True, conf=conf, iou=iou, classes=selected_classes)
        
        self._thread_safe_infer = thread_safe_infer

    def set_selected_classes(self, class_indices):
        """Set the selected classes for inference."""
        self.selected_classes = class_indices

    def infer(self, frame, conf, iou):
        """Run inference on a single frame with the current model."""
        if self.model is None or self._thread_safe_infer is None:
            return None
        # Use the thread-safe decorated function
        return self._thread_safe_infer(frame, conf, iou, self.selected_classes)

    def count_objects_in_regions(self, results, region_polygons):
        """Count objects in each region based on inference results."""
        # Get the region counts
        region_counts = [0 for _ in region_polygons]
        
        # Check if results are valid and have boxes
        if results and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            # Iterate through the boxes and count them in regions
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            for box, cls in zip(boxes, clss):
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                for idx, poly in enumerate(region_polygons):
                    if poly.contains(Point(center)):
                        region_counts[idx] += 1
                        
        return region_counts
    

class InferenceThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, list, object)  # Add object parameter for results
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self._pause = False
        self._stop = False
        self.current_frame_num = 0
        self.cap = None
        self.inference_engine = None
        self.region_polygons = []
        self.thresholds = {}

    def setup(self, video_path, inference_engine, regions, thresholds):
        self.cap = cv2.VideoCapture(video_path)
        self.inference_engine = inference_engine
        self.region_polygons = regions
        self.thresholds = thresholds
        self.current_frame_num = 0

    def run(self):
        while not self._stop and self.cap.isOpened():
            self.mutex.lock()
            if self._pause:
                self.condition.wait(self.mutex)
            self.mutex.unlock()

            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            results = self.inference_engine.infer(frame,
                self.thresholds['conf'],
                self.thresholds['iou'])
           
            region_counts = self.inference_engine.count_objects_in_regions(
                results, self.region_polygons)
           
            # Emit processed frame with results
            self.frame_processed.emit(frame, region_counts, results)
            self.current_frame_num += 1

        self.cap.release()
        self.finished.emit()

    def pause(self):
        self.mutex.lock()
        self._pause = True
        self.mutex.unlock()

    def resume(self):
        self.mutex.lock()
        self._pause = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def stop(self):
        self._stop = True
        self.resume()

    def seek(self, frame_num):
        self.mutex.lock()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame_num = frame_num
        self.mutex.unlock()
