import os 
from datetime import datetime

import cv2
import numpy as np
import supervision as sv

from shapely.geometry import Polygon, Point

from ultralytics import YOLO

from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint, QThread, pyqtSignal, QMutex, QWaitCondition, QRect
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QSlider, QFileDialog, 
                             QWidget, QGridLayout, QListWidget, QListWidgetItem, 
                             QAbstractItemView, QFormLayout, QTabWidget, 
                             QComboBox, QCheckBox, QSpacerItem, QSizePolicy)

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoRegionWidget(QWidget):
    """Widget for displaying video, playback controls, and drawing/editing rectangular regions only."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Video frame and display
        self.frame = None
        self.pixmap = None
        self.regions = []  # List of QRect
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

        # Inference
        self.inference_enabled = False
        self.inference_engine = None
        self.region_polygons = []
        self.conf = 0.3
        self.iou = 0.2

        # Video output handling - simplified and fixed
        self.video_sink = None
        self.video_path = None
        self.output_dir = None
        self.output_path = None
        self.should_write_video = False

        self._setup_ui()
        self.disable_video_region()

    def _setup_ui(self):
        """Setup the user interface components."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addStretch(1)  # Video area will be drawn in paintEvent
        
        controls = QHBoxLayout()
        
        # Control buttons
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
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setFocusPolicy(Qt.NoFocus)
        controls.addWidget(self.stop_btn)

        # Seek slider
        controls.addSpacing(8)
        controls.addStretch(1)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek)
        self.seek_slider.setMinimumWidth(600)
        self.seek_slider.setMaximumWidth(1000)
        controls.addWidget(self.seek_slider)
        controls.addStretch(1)
        controls.addSpacing(8)

        # Set button sizes
        max_btn_size = 32
        for btn in [self.step_back_btn, self.play_btn, self.pause_btn, self.step_fwd_btn, self.stop_btn]:
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

        self.layout.addLayout(controls)
        
    def closeEvent(self, event):
        """Handle widget close event."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()
        super().closeEvent(event)
        
    def enable_video_region(self):
        """Enable all controls in the video region widget."""
        self.setEnabled(True)
        for widget in [self.step_back_btn, self.play_btn, self.pause_btn, 
                       self.step_fwd_btn, self.stop_btn, self.seek_slider, 
                       self.speed_dropdown, self.frame_label]:
            widget.setEnabled(True)

    def disable_video_region(self):
        """Disable all controls in the video region widget."""
        self.setEnabled(False)
        for widget in [self.step_back_btn, self.play_btn, self.pause_btn, 
                       self.step_fwd_btn, self.stop_btn, self.seek_slider, 
                       self.speed_dropdown, self.frame_label]:
            widget.setEnabled(False)
            
    def play_video(self):
        """Play the video from the current position."""
        if not self.is_playing and self.cap:
            # If we don't have an active video sink but should be recording, create a new one
            if self.output_dir and not self.should_write_video:
                self._prepare_new_video_output()
                
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
        self._cleanup_video_sink()
        print("Video recording stopped - ready for new recording on next play")
        
        # Reset to first frame
        if self.cap:
            self.seek(0)
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.seek_slider.setValue(0)
            self.update_frame_label()

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
            self.update()
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
        
    def set_inference_params(self, inference_engine, region_polygons, conf, iou):
        """Set inference parameters for the video region."""
        self.inference_engine = inference_engine
        self.region_polygons = region_polygons
        self.conf = conf
        self.iou = iou

    def enable_inference(self, enable: bool):
        """Enable or disable inference in the video region."""
        self.inference_enabled = enable

    def set_region_visibility(self, visible: bool):
        """Set the visibility of regions in the video."""
        self.show_regions = visible
        self.update()
        
    def undo_region(self):
        """Undo the last region action."""
        if self.undo_stack:
            self.redo_stack.append(list(self.regions))
            self.regions = self.undo_stack.pop()
            self.update()

    def redo_region(self):
        """Redo the last undone region action."""
        if self.redo_stack:
            self.undo_stack.append(list(self.regions))
            self.regions = self.redo_stack.pop()
            self.update()
        
    # Region drawing methods
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
        
    def _make_rect(self, p1, p2):
        """Return a QRect from two points."""
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        return QRect(left, top, right - left, bottom - top)
        
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
        # Clean up existing video capture
        if self.cap:
            self.cap.release()
        
        # Clean up existing video sink
        self._cleanup_video_sink()
            
        # Load new video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Could not open video at {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.seek_slider.setMaximum(self.total_frames - 1)
        self.current_frame_number = 0
        self.is_playing = False
        self.video_path = video_path
        self.output_dir = output_dir
        
        # Setup output video if output directory is provided
        if output_dir and os.path.exists(output_dir):
            self._setup_video_output(video_path, output_dir)
        else:
            self.should_write_video = False
        
        # Load first frame
        self.seek(0)
        self.update()
        self.update_frame_label()
        self.enable_video_region()
        
    def process_frame_for_inference(self, frame):
        """Process frame for inference if enabled."""
        if not self.inference_enabled or not self.inference_engine:
            return frame
            
        try:
            # Run inference on the current frame
            results = self.inference_engine.infer(frame, self.conf, self.iou)
            # Count objects in defined regions
            region_counts = self.inference_engine.count_objects_in_regions(results, self.region_polygons)
            # Draw results on the frame
            frame = self.draw_inference_results(frame, region_counts, results)
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
            # Process the frame for inference if enabled
            processed_frame = self.process_frame_for_inference(frame.copy())
            
            # Write to video sink
            self._write_frame_to_sink(processed_frame)
            
            # Update display
            self.current_frame = processed_frame
            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.update()
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_number)
            self.seek_slider.blockSignals(False)
            self.update_frame_label()

    def draw_inference_results(self, frame, region_counts, results):
        """Draw inference results on the video frame using supervision's BoxAnnotator."""
        if not results or len(results) == 0:
            return frame
            
        try:
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Prepare labels for each detection
            class_names = []
            for cls in detections.class_id:
                idx = int(cls)
                if hasattr(self.parent, 'inference_engine') and self.parent.inference_engine:
                    if idx < len(self.parent.inference_engine.class_names):
                        class_names.append(self.parent.inference_engine.class_names[idx])
                    else:
                        class_names.append(str(idx))
                else:
                    class_names.append(str(idx))
                    
            confidences = detections.confidence
            labels = [f"{name}: {conf:.2f}" for name, conf in zip(class_names, confidences)]
            
            # Apply annotations
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)
            
            # Get selected annotators from parent if available
            selected_annotators = []
            if hasattr(self.parent, 'get_selected_annotators'):
                selected_annotators = self.parent.get_selected_annotators()
            else:
                selected_annotators = ["BoxAnnotator"]  # Default
                
            annotators = []
            for key in selected_annotators:
                if key == "BoxAnnotator":
                    annotators.append(sv.BoxAnnotator())
                elif key == "BoxCornerAnnotator":
                    annotators.append(sv.BoxCornerAnnotator())
                elif key == "DotAnnotator":
                    annotators.append(sv.DotAnnotator())
                elif key == "HaloAnnotator":
                    annotators.append(sv.HaloAnnotator())
                elif key == "PercentageBarAnnotator":
                    annotators.append(sv.PercentageBarAnnotator())
                elif key == "MaskAnnotator":
                    annotators.append(sv.MaskAnnotator())
                elif key == "PolygonAnnotator":
                    annotators.append(sv.PolygonAnnotator())
                    
            for annotator in annotators:
                frame = annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
        except Exception as e:
            print(f"Supervision annotate failed: {e}")
        
        # Draw region polygons
        region_polygons = getattr(self.parent, 'region_polygons', []) if self.parent else self.region_polygons
        for idx, poly in enumerate(region_polygons):
            if idx < len(region_counts):
                pts = np.array(list(poly.exterior.coords), np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                centroid = poly.centroid
                cv2.putText(frame, str(region_counts[idx]), (int(centroid.x), int(centroid.y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self._cleanup_video_sink()
        if self.cap:
            self.cap.release()
            

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
        self.model = YOLO(model_path, task=task)
        self.class_names = list(self.model.names.values())
        
        # Decorate the infer method for thread safety
        @ThreadingLocked()
        def thread_safe_infer(frame, conf, iou, selected_classes):
            return self.model.track(frame, 
                                    persist=True, 
                                    conf=conf, 
                                    iou=iou, 
                                    classes=selected_classes,
                                    half=True,
                                    retina_masks=task == "segment")
        
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
        if not region_polygons:
            return []
        
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
        
        self.task = None                            # Task parameter, modified in subclasses
        
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        
        self.video_region_widget = None             # Initialized in setup_video_layout
        self.inference_engine = InferenceEngine()

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
        
    def closeEvent(self, event):
        """Ensure inference thread is stopped before closing the dialog."""
        self.close_video_sink()
        super().closeEvent(event)

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

        # Class filter
        self.class_filter_widget = QListWidget()
        self.class_filter_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        form_layout.addRow(QLabel("Class Filter:"), self.class_filter_widget)

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
        self.iou_thresh_slider = QSlider(Qt.Horizontal)
        self.iou_thresh_slider.setRange(0, 100)
        self.iou_thresh_slider.setValue(int(self.iou_thresh * 100))
        self.iou_thresh_slider.valueChanged.connect(self.update_iou_label)
        form_layout.addRow(QLabel("IoU Threshold:"), self.iou_thresh_slider)

        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        form_layout.addRow(QLabel("Uncertainty Threshold:"), self.uncertainty_thresh_slider)

        self.area_threshold_min_slider = QSlider(Qt.Horizontal)
        self.area_threshold_min_slider.setRange(0, 100)
        self.area_threshold_min_slider.setValue(int(self.area_thresh_min * 100))
        self.area_threshold_min_slider.valueChanged.connect(self.update_area_label)
        form_layout.addRow(QLabel("Area Threshold Min:"), self.area_threshold_min_slider)

        self.area_threshold_max_slider = QSlider(Qt.Horizontal)
        self.area_threshold_max_slider.setRange(0, 100)
        self.area_threshold_max_slider.setValue(int(self.area_thresh_max * 100))
        self.area_threshold_max_slider.valueChanged.connect(self.update_area_label)
        form_layout.addRow(QLabel("Area Threshold Max:"), self.area_threshold_max_slider)

        group_box.setLayout(form_layout)
        self.controls_layout.addWidget(group_box)

    def setup_video_layout(self):
        """Setup the video region widget inside a group box (no tabs)."""
        group_box = QGroupBox("Video Player")
        vbox = QVBoxLayout()
        
        self.video_region_widget = VideoRegionWidget(self)
        vbox.addWidget(self.video_region_widget)
        
        group_box.setLayout(vbox)
        self.video_layout.addWidget(group_box)

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

    def setup_annotators_layout(self):
        """Setup the annotator selection layout using a QListWidget with checkable items."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def setup_inference_layout(self):
        """Setup the inference control group with Enable/Disable buttons."""
        group_box = QGroupBox("Inference Controls")
        layout = QHBoxLayout()
        
        self.enable_inference_btn = QPushButton("Enable Inference")
        self.enable_inference_btn.clicked.connect(self.enable_inference)
        # self.enable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
        layout.addWidget(self.enable_inference_btn)
        
        self.disable_inference_btn = QPushButton("Disable Inference")
        self.disable_inference_btn.clicked.connect(self.disable_inference)
        # self.disable_inference_btn.setFocusPolicy(Qt.NoFocus)  # Prevent focus/highlighting
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
        
        # Update inference params
        self.video_region_widget.set_inference_params(
            self.inference_engine,
            getattr(self, 'region_polygons', []),
            self.uncertainty_thresh,
            self.iou_thresh
        )

    def update_iou_label(self, value):
        """Update IoU threshold and label"""
        value = value / 100.0
        self.iou_thresh = value
        self.main_window.update_iou_thresh(value)
        self.iou_thresh_label.setText(f"{value:.2f}")
        
        # Update inference params
        self.video_region_widget.set_inference_params(
            self.inference_engine,
            getattr(self, 'region_polygons', []),
            self.uncertainty_thresh,
            self.iou_thresh
        )

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
        
        # Update inference params
        self.video_region_widget.set_inference_params(
            self.inference_engine,
            getattr(self, 'region_polygons', []),
            self.uncertainty_thresh,
            self.iou_thresh
        )

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
        
    def clear_regions(self):
        """Clear all regions from the video region widget and update display."""
        self.video_region_widget.regions.clear()
        self.video_region_widget.update()

    def enable_inference(self):
        """Enable inference on the video region."""
        if not self.model_path or not self.video_path:
            return
        
        # Load the model and set inference parameters
        self.region_polygons = [
            Polygon([
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.right(), rect.bottom()),
                (rect.left(), rect.bottom())
            ]) for rect in self.video_region_widget.regions
        ]
        # Set inference parameters in the video region widget
        self.video_region_widget.set_inference_params(
            self.inference_engine,
            self.region_polygons,
            self.uncertainty_thresh,
            self.iou_thresh
        )
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(True)
        self.enable_inference_btn.setEnabled(False)
        self.disable_inference_btn.setEnabled(True)

        # Refresh the current frame without inference
        self.video_region_widget.seek(self.video_region_widget.current_frame_number)

    def disable_inference(self):
        """Disable inference on the video region."""
        # Only set the flag and update UI, do not change video state
        self.video_region_widget.enable_inference(False)
        self.enable_inference_btn.setEnabled(True)
        self.disable_inference_btn.setEnabled(False)
        
        # Refresh the current frame without inference
        self.video_region_widget.seek(self.video_region_widget.current_frame_number)

    def close_video_sink(self):
        if self.video_sink is not None:
            self.video_sink.__exit__(None, None, None)
            self.video_sink = None
