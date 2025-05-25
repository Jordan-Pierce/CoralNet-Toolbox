import os

import cv2
import numpy as np

from shapely.geometry import Polygon, Point

from ultralytics import YOLO

from PyQt5.QtCore import Qt, QRect, QPoint, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, 
                             QSpinBox, QFrame, QWidget, QGridLayout, QListWidget, 
                             QListWidgetItem, QAbstractItemView, QFormLayout)

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
    

class InferenceEngine:
    """Handles model loading, inference, and class filtering."""
    def __init__(self):
        self.model = None
        self.class_names = []
        self.selected_classes = []

    def load_model(self, model_path, task):
        self.model = YOLO(model_path, task=task if task else None)
        self.class_names = list(self.model.names.values())

    def set_selected_classes(self, class_indices):
        self.selected_classes = class_indices

    def infer(self, frame, conf, iou):
        if self.model is None:
            return None
        
        # Run inference on the frame
        return self.model.track(frame, persist=True, conf=conf, iou=iou, classes=self.selected_classes)

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


class VideoPlayerWidget(QWidget):
    """Widget for video playback and frame display with controls."""
    def __init__(self, parent=None):
        """Initialize the video player widget."""
        super().__init__(parent)
        self.cap = None  # OpenCV VideoCapture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.current_frame = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        self.init_ui()

    def init_ui(self):
        """Set up the UI layout and controls."""
        layout = QVBoxLayout(self)
        
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_label)
        
        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek)
        
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.seek_slider)
        
        layout.addLayout(controls)

    def load_video(self, video_path):
        """Load a video file and prepare for playback."""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.seek_slider.setMaximum(self.total_frames - 1)
        self.current_frame_number = 0
        self.is_playing = False
        self.seek(0)

    def on_frame_update(self, frame, frame_number):
        """Update the displayed frame and slider position."""
        self.current_frame = frame
        self.current_frame_number = frame_number
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.frame_label.setPixmap(pixmap.scaled(self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.seek_slider.setValue(frame_number)

    def next_frame(self):
        """Advance to the next frame in the video and update the display."""
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Update current frame number (OpenCV is 0-based)
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                h, w, c = frame.shape
                # Convert frame to QImage and display
                q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.frame_label.setPixmap(pixmap.scaled(self.frame_label.size(), 
                                                         Qt.KeepAspectRatio, 
                                                         Qt.SmoothTransformation))
                # Update slider without triggering signals
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(self.current_frame_number)
                self.seek_slider.blockSignals(False)
            else:
                # Stop playback if no more frames
                self.timer.stop()
                self.is_playing = False

    def toggle_play(self):
        """Start video playback."""
        if self.cap:
            self.is_playing = True
            interval = int(1000 / self.fps)
            self.timer.start(interval)

    def toggle_pause(self):
        """Pause video playback."""
        self.is_playing = False
        self.timer.stop()

    def seek(self, frame_number):
        """Seek to a specific frame in the video."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_number = frame_number
                h, w, c = frame.shape
                # Convert frame to QImage and display
                q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.frame_label.setPixmap(pixmap.scaled(self.frame_label.size(), 
                                                         Qt.KeepAspectRatio, 
                                                         Qt.SmoothTransformation))


class VideoRegionWidget(QWidget):
    """Widget for displaying video and drawing/editing regions."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.frame = None
        self.pixmap = None
        self.regions = []
        self.drawing = False
        self.current_polygon = []
        self.selected_region = None
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)

    def load_video(self, path):
        """Load the first frame of the video for preview and region drawing."""
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.frame = frame
            self.update_pixmap()
            self.update()

    def update_pixmap(self):
        """Update the QPixmap from the current frame."""
        if self.frame is not None:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qimg)

    def paintEvent(self, event):
        """Draw the video frame and regions."""
        painter = QPainter(self)
        if self.pixmap:
            scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled)
            
        # Draw regions
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        for region in self.regions:
            if len(region) > 1:
                painter.drawPolygon(region)
                
        # Draw current polygon
        if self.drawing and len(self.current_polygon) > 1:
            painter.drawPolyline(self.current_polygon)

    def mousePressEvent(self, event):
        """Handle mouse press for region drawing."""
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            if not self.drawing:
                self.drawing = True
                self.current_polygon = [pos]
            else:
                self.current_polygon.append(pos)
            self.update()
            
        elif event.button() == Qt.RightButton and self.drawing:
            # Finish polygon
            if len(self.current_polygon) > 2:
                self.regions.append(self.current_polygon[:])
            self.drawing = False
            self.current_polygon = []
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move for live region drawing."""
        if self.drawing and self.current_polygon:
            # Replace last point instead of appending
            if len(self.current_polygon) > 0:
                self.current_polygon[-1] = event.pos()
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Finish polygon on double click."""
        if self.drawing and len(self.current_polygon) > 2:
            self.regions.append(self.current_polygon[:])
            self.drawing = False
            self.current_polygon = []
            self.update()

    def display_frame(self, frame):
        """Display a processed frame in the video widget."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.update()


class Base(QDialog):
    """Dialog for video inference with region selection and parameter controls."""
    def __init__(self, main_window, parent=None):
        """Initialize the Video Inference dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Video Inference")
        self.showMaximized()  # Open in maximized view
        
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
        self.setup_model_layout()  # Only model path input
        # Setup the parameters layout
        self.setup_parameters_layout()  # Sliders for thresholds
        # Add a stretch
        self.controls_layout.addStretch()
        # Setup the video player widget
        self.setup_video_group()
        # Initialize threshold sliders with main_window values
        self.initialize_thresholds()
        # Setup Run/Cancel buttons
        self.setup_buttons_layout()

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
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_conf_label)
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
        
        # Class filter widget
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addRow(QLabel("Class Filter:"), self.class_filter_widget)
        
        group_box.setLayout(layout)
        self.controls_layout.addWidget(group_box)

    def setup_video_group(self):
        """Setup the video player widget and region widget."""
        # Add both video player and region widget
        self.video_player = VideoPlayerWidget(self)
        self.video_region_widget = VideoRegionWidget(self)
        
        self.video_layout.addWidget(self.video_player)
        self.video_layout.addWidget(self.video_region_widget)

    def setup_buttons_layout(self):
        """Setup the Run and Cancel buttons at the bottom of the controls layout."""
        btn_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_inference_on_video)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        self.controls_layout.addLayout(btn_layout)

    def initialize_thresholds(self):
        """Initialize all threshold sliders with current values."""
        self.initialize_uncertainty_thresh()
        self.initialize_iou_thresh()
        self.initialize_area_threshold()

    def initialize_uncertainty_thresh(self):
        """Initialize the confidence threshold slider with the current value"""
        current_value = getattr(self.main_window, 'get_uncertainty_thresh', lambda: 0.25)()
        self.uncertainty_thresh_slider.setValue(int(current_value * 100))
        self.uncertainty_thresh = current_value

    def initialize_iou_thresh(self):
        """Initialize the IOU threshold slider with the current value"""
        current_value = getattr(self.main_window, 'get_iou_thresh', lambda: 0.45)()
        self.iou_thresh_slider.setValue(int(current_value * 100))
        self.iou_thresh = current_value

    def initialize_area_threshold(self):
        """Initialize the area threshold range slider"""
        current_min, current_max = getattr(self.main_window, 'get_area_thresholds', lambda: (0, 1))()
        self.area_threshold_min_slider.setValue(int(current_min * 100))
        self.area_threshold_max_slider.setValue(int(current_max * 100))
        self.area_thresh_min = current_min
        self.area_thresh_max = current_max

    def update_conf_label(self, value):
        """Update confidence threshold and label"""
        value = value / 100.0
        self.uncertainty_thresh = value
        if hasattr(self.main_window, 'update_uncertainty_thresh'):
            self.main_window.update_uncertainty_thresh(value)
        self.uncertainty_thresh_label.setText(f"{value:.2f}")

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        if hasattr(self.main_window, 'update_iou_thresh'):
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
        if hasattr(self.main_window, 'update_area_thresholds'):
            self.main_window.update_area_thresholds(self.area_thresh_min, self.area_thresh_max)
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
            self.video_player.load_video(file_name)
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
            item.setCheckState(Qt.Unchecked)
            self.class_filter_widget.addItem(item)
        self.class_filter_widget.itemChanged.connect(self.update_selected_classes)

    def update_selected_classes(self):
        """Update the selected classes based on the class filter widget."""
        selected = []
        for i in range(self.class_filter_widget.count()):
            item = self.class_filter_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        self.inference_engine.set_selected_classes(selected)

    def run_inference_on_video(self):
        """Run YOLOv8 inference on the selected video, counting objects in user-defined regions."""
        # Check if model and video paths are set
        if not self.model_path or not self.video_path:
            return
        
        # Ensure model is loaded
        if self.inference_engine.model is None:
            self.inference_engine.load_model(self.model_path, task=self.task)
            
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Prepare region polygons from the region widget
        region_polygons = [Polygon([(pt.x(), pt.y()) for pt in region]) for region in self.video_region_widget.regions]
        
        # Iterate through frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference using InferenceEngine
            results = self.inference_engine.infer(frame, self.uncertainty_thresh, self.iou_thresh)
            
            # Count objects in regions
            region_counts = self.inference_engine.count_objects_in_regions(results, region_polygons)
            
            # Draw bounding boxes if results exist
            if results and hasattr(results[0].boxes, 'xyxy'):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    
            # Draw regions and counts
            for idx, poly in enumerate(region_polygons):
                pts = np.array(list(poly.exterior.coords), np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                centroid = poly.centroid
                cv2.putText(frame, str(region_counts[idx]), (int(centroid.x), int(centroid.y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            # Update video widget
            self.video_region_widget.display_frame(frame)
            
            # Optional: add a delay for real-time display
            cv2.waitKey(int(1000 / fps))
            
        cap.release()
