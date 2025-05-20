import warnings
import os
import math
import gc
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QApplication, QMessageBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
                             QStyle, QFrame, QTabWidget, QWidget)
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Video Player Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoPlayerThread(QThread):
    """Thread for playing video frames without blocking the UI"""
    update_frame = pyqtSignal(object, int)
    
    def __init__(self, video_path, fps=30):
        super().__init__()
        self.video_path = video_path
        self.fps = fps
        self.running = False
        self.paused = False
        self.mutex = QMutex()
        self.current_frame_number = 0
        self.cap = None
        self.total_frames = 0
        
    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.running = True
        
        while self.running:
            if not self.paused:
                self.mutex.lock()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                ret, frame = self.cap.read()
                self.mutex.unlock()
                
                if ret:
                    # Convert frame to RGB (from BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.update_frame.emit(frame_rgb, self.current_frame_number)
                    
                    # Move to next frame
                    self.current_frame_number = (self.current_frame_number + 1) % self.total_frames
                else:
                    # Could not read frame, maybe end of video
                    self.running = False
                    
                # Control playback speed
                time.sleep(1 / self.fps)
            else:
                # If paused, just sleep a bit to avoid CPU overuse
                time.sleep(0.1)
                
        # Clean up
        if self.cap:
            self.cap.release()
    
    def seek(self, frame_number):
        """Seek to a specific frame"""
        self.mutex.lock()
        self.current_frame_number = min(max(0, frame_number), self.total_frames - 1)
        self.mutex.unlock()
        
        # If paused, immediately show the frame
        if self.paused:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_frame.emit(frame_rgb, self.current_frame_number)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()


# ----------------------------------------------------------------------------------------------------------------------
# Frame Extraction Worker
# ----------------------------------------------------------------------------------------------------------------------


class FrameExtractorThread(QThread):
    """Thread for extracting frames without blocking the UI"""
    progress_updated = pyqtSignal(int)
    extraction_completed = pyqtSignal(list)
    extraction_error = pyqtSignal(str)
    
    def __init__(self, video_file, output_dir, frame_prefix, ext, frame_indices):
        super().__init__()
        self.video_file = video_file
        self.output_dir = output_dir
        self.frame_prefix = frame_prefix
        self.ext = ext
        self.frame_indices = frame_indices
        self.frame_paths = []
    
    def run(self):
        """
        Extracts video frames either in linear mode, emitting progress and completion signals.
        """
        try:
            # Linear extraction (single-threaded)
            extracted_paths = []
            progress_count = 0
            if self.frame_indices == sorted(self.frame_indices):
                # Use a single VideoCapture for ordered extraction
                cap = cv2.VideoCapture(self.video_file)
                if not cap.isOpened():
                    self.extraction_error.emit(f"Failed to open video file: {self.video_file}")
                    return
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                last_pos = None
                for frame_idx in self.frame_indices:
                    if frame_idx < 0 or frame_idx >= total_frames:
                        print(f"[ERROR] Frame index {frame_idx} is out of range (0, {total_frames-1})")
                        continue
                    if last_pos is None or frame_idx != last_pos + 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    last_pos = frame_idx
                    if not ret or frame is None:
                        print(f"[ERROR] Failed to read frame {frame_idx}")
                        continue
                    output_path = f"{self.output_dir}/{self.frame_prefix}_{frame_idx}.{self.ext}"
                    output_dir = os.path.dirname(output_path)
                    if output_dir and not os.path.exists(output_dir):
                        try:
                            os.makedirs(output_dir, exist_ok=True)
                        except Exception as e:
                            print(f"[ERROR] Could not create output directory {output_dir}: {e}")
                            continue
                    if output_path.lower().endswith(('.jpg', '.jpeg')):
                        success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    elif output_path.lower().endswith('.png'):
                        success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    else:
                        success = cv2.imwrite(output_path, frame)
                    if not success:
                        print(f"[ERROR] Failed to write frame to {output_path}")
                        continue
                    extracted_paths.append(output_path)
                    progress_count += 1
                    self.progress_updated.emit(progress_count)
                cap.release()
                self.frame_paths = extracted_paths
                self.extraction_completed.emit(extracted_paths)
            else:
                # Fallback: open/close per frame (existing logic)
                for frame_idx in self.frame_indices:
                    frame_path = f"{self.output_dir}/{self.frame_prefix}_{frame_idx}.{self.ext}"
                    result = self.extract_single_frame(self.video_file, frame_idx, frame_path)
                    if result:
                        extracted_paths.append(result)
                    progress_count += 1
                    self.progress_updated.emit(progress_count)
                self.frame_paths = extracted_paths
                self.extraction_completed.emit(extracted_paths)
        except Exception as e:
            # Emit error signal if any exception occurs
            self.extraction_error.emit(str(e))
    
    @staticmethod
    def extract_single_frame(video_file, frame_index, output_path):
        """Extract a single frame and save it to the output path, with debug checks."""
        print(f"[DEBUG] Attempting to extract frame {frame_index} from {video_file} to {output_path}")
    
        if not os.path.exists(video_file):
            print(f"[ERROR] Video file does not exist: {video_file}")
            return None
    
        local_cap = cv2.VideoCapture(video_file)
        if not local_cap.isOpened():
            print(f"[ERROR] Failed to open video file: {video_file}")
            return None
    
        total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[DEBUG] Video has {total_frames} frames")
    
        if frame_index < 0 or frame_index >= total_frames:
            print(f"[ERROR] Frame index {frame_index} is out of range (0, {total_frames-1})")
            local_cap.release()
            return None
    
        local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = local_cap.read()
        local_cap.release()
    
        if not ret or frame is None:
            print(f"[ERROR] Failed to read frame {frame_index}")
            return None
    
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"[DEBUG] Created output directory: {output_dir}")
            except Exception as e:
                print(f"[ERROR] Could not create output directory {output_dir}: {e}")
                return None
    
        # Write frame to file
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[DEBUG] Writing JPEG frame to {output_path}: {'Success' if success else 'Failed'}")
        elif output_path.lower().endswith('.png'):
            success = cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"[DEBUG] Writing PNG frame to {output_path}: {'Success' if success else 'Failed'}")
        else:
            success = cv2.imwrite(output_path, frame)
            print(f"[DEBUG] Writing frame to {output_path}: {'Success' if success else 'Failed'}")
    
        if not success:
            print(f"[ERROR] Failed to write frame to {output_path}")
            return None
    
        print(f"[DEBUG] Successfully extracted frame {frame_index} to {output_path}")
        return output_path


# ----------------------------------------------------------------------------------------------------------------------
# Frame Cache Manager
# ----------------------------------------------------------------------------------------------------------------------


class FrameCache:
    """Cache for storing video frames to avoid repeated disk reads"""
    def __init__(self, max_size=30):
        self.cache = {}  # frame_idx -> frame data
        self.max_size = max_size
        self.access_order = []  # LRU tracking
        
    def get(self, frame_idx):
        """Get a frame from the cache or None if not present"""
        if frame_idx in self.cache:
            # Update access order (move to end as most recently used)
            self.access_order.remove(frame_idx)
            self.access_order.append(frame_idx)
            return self.cache[frame_idx]
        return None
        
    def put(self, frame_idx, frame):
        """Add a frame to the cache"""
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size and self.access_order:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            
        # Add new item
        self.cache[frame_idx] = frame
        self.access_order.append(frame_idx)
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()


# ----------------------------------------------------------------------------------------------------------------------
# Main Import Frames Dialog
# ----------------------------------------------------------------------------------------------------------------------

class ImportFrames(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Import Frames from Video")
        self.resize(1000, 600)

        self.video_file = ""
        self.output_dir = ""
        self.num_frames = 0
        self.start_frame = 0
        self.end_frame = 0

        self.frame_paths = []
        self.frame_cache = FrameCache(max_size=30)  # Cache 30 frames max
        
        # Video player elements
        self.player_thread = None
        self.is_playing = False
        self.playback_timer = None

        # Main horizontal layout to hold controls and preview
        main_layout = QHBoxLayout()

        # Left side - Controls
        controls_layout = QVBoxLayout()

        # Move existing layouts to controls
        controls_layout.addWidget(self.create_info_group())
        controls_layout.addWidget(self.create_import_group())
        controls_layout.addWidget(self.create_output_group())
        controls_layout.addWidget(self.create_sample_group())
        controls_layout.addLayout(self.create_buttons_layout())

        # Right side - Preview
        preview_layout = self.setup_preview_layout()

        # Add both sides to main layout
        main_layout.addLayout(controls_layout, stretch=60)
        main_layout.addLayout(preview_layout, stretch=40)

        self.setLayout(main_layout)

        self.cap = None
        self.current_frame_idx = 0
        
        # Initialize timer for UI updates
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(500)  # Update UI every 500ms

    def create_info_group(self):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a video file to extract frames from.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        return group_box

    def create_import_group(self):
        group_box = QGroupBox("Import")
        layout = QFormLayout()

        # Video file selection
        self.video_file_edit = QLineEdit()
        self.video_file_button = QPushButton("Browse...")
        self.video_file_button.clicked.connect(self.browse_video_file)
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_file_edit)
        video_layout.addWidget(self.video_file_button)
        layout.addRow("Video File:", video_layout)

        group_box.setLayout(layout)
        return group_box

    def create_output_group(self):
        group_box = QGroupBox("Output")
        layout = QFormLayout()

        # Output directory row
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        # Frame prefix row
        prefix_layout = QHBoxLayout()
        self.frame_prefix_edit = QLineEdit()
        self.frame_prefix_edit.setPlaceholderText("frame_prefix_(idx).ext")
        prefix_layout.addWidget(self.frame_prefix_edit)
        layout.addRow("Frame Prefix:", prefix_layout)

        # Extension row
        ext_layout = QHBoxLayout()
        self.frame_ext_combo = QComboBox()
        self.frame_ext_combo.setEditable(True)
        self.frame_ext_combo.addItems(['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
        ext_layout.addWidget(self.frame_ext_combo)
        layout.addRow("Frame Extension:", ext_layout)

        group_box.setLayout(layout)
        return group_box

    def create_sample_group(self):
        group_box = QGroupBox("Sample Frames")
        layout = QVBoxLayout()

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.current_tab = "range"  # Track active tab

        # Create Range Tab
        range_tab = QWidget()
        range_layout = QFormLayout()

        # Initialize time_range_label
        self.time_range_label = QLabel("Time Range: No video loaded")

        # Range slider for selecting frames
        self.range_start_slider = QSlider(Qt.Horizontal)
        self.range_start_slider.setEnabled(False)
        self.range_start_slider.setTickPosition(QSlider.TicksBelow)
        self.range_start_slider.setTickInterval(10)
        self.range_start_slider.valueChanged.connect(self.update_range_slider_label)
        self.range_start_slider.valueChanged.connect(self.update_calculated_frames)

        self.range_end_slider = QSlider(Qt.Horizontal)
        self.range_end_slider.setEnabled(False)
        self.range_end_slider.setTickPosition(QSlider.TicksBelow)
        self.range_end_slider.setTickInterval(10)
        self.range_end_slider.valueChanged.connect(self.update_range_slider_label)
        self.range_end_slider.valueChanged.connect(self.update_calculated_frames)

        # Create range input layout using spinboxes instead of line edits
        range_input_layout = QHBoxLayout()
        self.range_start_spinbox = QSpinBox()
        self.range_start_spinbox.setFixedWidth(80)
        self.range_start_spinbox.setAlignment(Qt.AlignCenter)
        self.range_start_spinbox.valueChanged.connect(self.range_spinbox_changed)

        self.range_end_spinbox = QSpinBox()
        self.range_end_spinbox.setFixedWidth(80)
        self.range_end_spinbox.setAlignment(Qt.AlignCenter)
        self.range_end_spinbox.valueChanged.connect(self.range_spinbox_changed)

        range_input_layout.addWidget(QLabel("Frame Range:"))
        range_input_layout.addWidget(self.range_start_spinbox)
        range_input_layout.addWidget(QLabel("-"))
        range_input_layout.addWidget(self.range_end_spinbox)
        range_input_layout.addStretch()

        # Every N frames to sample
        self.every_n_frames_spinbox = QSpinBox()
        self.every_n_frames_spinbox.setRange(1, 10000000)
        self.every_n_frames_spinbox.setValue(24)
        self.every_n_frames_spinbox.valueChanged.connect(self.update_calculated_frames)

        range_layout.addRow("Sample Every N Frames:", self.every_n_frames_spinbox)
        range_layout.addRow("Select Start Frame:", self.range_start_slider)
        range_layout.addRow("Select End Frame:", self.range_end_slider)
        range_layout.addRow("", range_input_layout)
        range_layout.addRow("", self.time_range_label)
        range_tab.setLayout(range_layout)

        # Create Specific Frames Tab
        specific_tab = QWidget()
        specific_layout = QFormLayout()

        # Specific frames to extract
        self.specific_frames_edit = QLineEdit()
        self.specific_frames_edit.setPlaceholderText("e.g., 1, 2, 3, 5-10, 15")
        self.specif_frames_button = QPushButton()
        self.specif_frames_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.specif_frames_button.clicked.connect(self.update_calculated_frames)
        specific_frames_layout = QHBoxLayout()
        specific_frames_layout.addWidget(self.specific_frames_edit)
        specific_frames_layout.addWidget(self.specif_frames_button)

        specific_layout.addRow("Enter Frame Numbers:", specific_frames_layout)
        specific_tab.setLayout(specific_layout)

        # Add tabs to widget
        self.tab_widget.addTab(range_tab, "Frame Range")
        self.tab_widget.addTab(specific_tab, "Specific Frames")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Add tab widget to layout
        layout.addWidget(self.tab_widget)

        # Calculated frames display
        calc_frames_layout = QHBoxLayout()
        calc_frames_label = QLabel("Calculated Frames:")
        self.calculated_frames_edit = QLineEdit()
        self.calculated_frames_edit.setReadOnly(True)
        self.calculated_frames_edit.setText("Load a video to calculate frames")

        calc_frames_layout.addWidget(calc_frames_label)
        calc_frames_layout.addWidget(self.calculated_frames_edit)
        layout.addLayout(calc_frames_layout)

        group_box.setLayout(layout)
        return group_box

    def on_tab_changed(self, index):
        """Handle tab changes"""
        self.current_tab = "range" if index == 0 else "specific"
        self.update_calculated_frames()

    def create_buttons_layout(self):
        buttons_layout = QHBoxLayout()

        self.extract_button = QPushButton("Extract")
        self.extract_import_button = QPushButton("Extract and Import")
        self.cancel_button = QPushButton("Cancel")

        self.extract_button.clicked.connect(lambda: self.import_frames(import_after=False))
        self.extract_import_button.clicked.connect(lambda: self.import_frames(import_after=True))
        self.cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.extract_button)
        buttons_layout.addWidget(self.extract_import_button)
        buttons_layout.addWidget(self.cancel_button)

        return buttons_layout

    def setup_preview_layout(self):
        """Set up the video preview panel with video player capabilities"""
        preview_layout = QVBoxLayout()

        # Preview group box
        preview_group = QGroupBox("Video Preview")
        preview_inner_layout = QVBoxLayout()

        # Frame display
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("QLabel { background-color: black; }")
        preview_inner_layout.addWidget(self.preview_label)

        # Video control layout
        vc_layout = QHBoxLayout()
        
        # Add play/pause button
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)
        
        # Add previous/next frame buttons
        self.prev_frame_btn = QPushButton("←")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)

        self.next_frame_btn = QPushButton("→")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        
        # Add frame number input
        self.frame_number_spinbox = QSpinBox()
        self.frame_number_spinbox.setFixedWidth(80)
        self.frame_number_spinbox.setAlignment(Qt.AlignCenter)
        self.frame_number_spinbox.setEnabled(False)
        
        # Add go-to-frame button
        self.goto_frame_btn = QPushButton()
        self.goto_frame_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.goto_frame_btn.clicked.connect(self.goto_frame)
        self.goto_frame_btn.setEnabled(False)
        
        # Add controls to layout
        vc_layout.addWidget(self.play_pause_btn)
        vc_layout.addWidget(self.prev_frame_btn)
        vc_layout.addWidget(self.next_frame_btn)
        vc_layout.addWidget(self.frame_number_spinbox)
        vc_layout.addWidget(self.goto_frame_btn)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        
        # Add layouts to preview group
        preview_inner_layout.addLayout(vc_layout)
        preview_inner_layout.addWidget(self.frame_slider)
        preview_group.setLayout(preview_inner_layout)
        preview_layout.addWidget(preview_group)

        # Frame info group box
        info_group = QGroupBox("Frame Information")
        info_layout = QVBoxLayout()

        # Frame counter and timestamp with some styling
        self.frame_counter = QLabel("Frame: 0 / 0")
        self.frame_counter.setAlignment(Qt.AlignCenter)
        self.frame_counter.setStyleSheet("QLabel { font-size: 11pt; }")

        self.frame_timestamp = QLabel("Time: 00:00")
        self.frame_timestamp.setAlignment(Qt.AlignCenter)
        self.frame_timestamp.setStyleSheet("QLabel { font-size: 11pt; }")

        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        info_layout.addWidget(self.frame_counter)
        info_layout.addWidget(line)
        info_layout.addWidget(self.frame_timestamp)

        info_group.setLayout(info_layout)
        preview_layout.addWidget(info_group)

        return preview_layout

    def goto_frame(self):
        """Go to the frame specified in the frame number spinbox"""
        if not self.player_thread:
            return
            
        frame_num = self.frame_number_spinbox.value()
        self.current_frame_idx = frame_num
        self.frame_slider.setValue(frame_num)
        
        # If player exists, tell it to seek to this frame
        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.seek(frame_num)

    def toggle_playback(self):
        """Toggle video playback state"""
        if not self.player_thread:
            return
            
        if self.is_playing:
            # Pause the video
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.player_thread.toggle_pause()
        else:
            # Play the video
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.player_thread.toggle_pause()
            
        self.is_playing = not self.is_playing

    def update_ui_elements(self):
        """Update UI elements periodically"""
        if self.player_thread and self.player_thread.isRunning() and self.is_playing:
            # Update slider position with current frame
            current_frame = self.player_thread.current_frame_number
            self.frame_slider.setValue(current_frame)
            
            # Update displayed frame number and time
            if hasattr(self.player_thread, 'fps') and self.player_thread.fps > 0:
                total_frames = self.player_thread.total_frames
                self.frame_counter.setText(f"Frame: {current_frame} / {total_frames}")
                
                time_in_seconds = current_frame / self.player_thread.fps
                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)
                self.frame_timestamp.setText(f"Time: {minutes:02d}:{seconds:02d}")

    def on_frame_update(self, frame, frame_number):
        """Handle frame updates from the video player thread"""
        # Update the UI with the new frame
        if frame is None:
            return
            
        # Create QImage from numpy array
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        
        # Calculate scale to fit the preview area while maintaining aspect ratio
        label_size = self.preview_label.size()
        img_ratio = w / h
        label_ratio = label_size.width() / label_size.height()
        
        if img_ratio > label_ratio:  # Image is wider than label
            scaled_size = QSize(label_size.width(), int(label_size.width() / img_ratio))
        else:  # Image is taller than label
            scaled_size = QSize(int(label_size.height() * img_ratio), label_size.height())
            
        # Set the pixmap
        pixmap = QPixmap.fromImage(q_img)
        self.preview_label.setPixmap(pixmap.scaled(
            scaled_size.width(), scaled_size.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update frame information
        self.current_frame_idx = frame_number
        
        # Update frame number without triggering value changed event
        self.frame_number_spinbox.blockSignals(True)
        self.frame_number_spinbox.setValue(frame_number)
        self.frame_number_spinbox.blockSignals(False)

    def slider_changed(self, value):
        """Handle frame slider value changes"""
        self.current_frame_idx = value
        
        # If player exists, tell it to seek to this frame
        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.seek(value)
        
        # Update frame number without triggering value changed event
        self.frame_number_spinbox.blockSignals(True)
        self.frame_number_spinbox.setValue(value)
        self.frame_number_spinbox.blockSignals(False)

    def next_frame(self):
        """Show next frame"""
        if self.player_thread and self.player_thread.isRunning():
            next_frame = min(self.current_frame_idx + 1, self.player_thread.total_frames - 1)
            self.player_thread.seek(next_frame)
            self.frame_slider.setValue(next_frame)

    def prev_frame(self):
        """Show previous frame"""
        if self.player_thread and self.player_thread.isRunning():
            prev_frame = max(self.current_frame_idx - 1, 0)
            self.player_thread.seek(prev_frame)
            self.frame_slider.setValue(prev_frame)

    def browse_video_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Video File",
                                                   "",
                                                   "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
                                                   options=options)

        if file_name:
            if not os.path.exists(file_name):
                QMessageBox.warning(self,
                                    "Invalid Video File",
                                    "Please select a valid video file.")
                return
                
            # Set the video file path in the edit box
            self.video_file_edit.setText(file_name)
            
            # Make cursor busy
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # Stop any existing player
            self.stop_video_player()
            
            try:
                # Open video to get basic info
                cap = cv2.VideoCapture(file_name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Set video properties
                self.fps = fps
                self.total_frames = total_frames
                
                # Initialize the video player
                self.init_video_player(file_name, fps)
                
                # Update the range slider
                self.update_range_slider()
                
                # Update frame slider
                self.frame_slider.setEnabled(True)
                self.frame_slider.setRange(0, total_frames - 1)
                self.frame_slider.setValue(0)
                
                # Enable controls
                self.play_pause_btn.setEnabled(True)
                self.prev_frame_btn.setEnabled(True)
                self.next_frame_btn.setEnabled(True)
                self.frame_number_spinbox.setEnabled(True)
                self.goto_frame_btn.setEnabled(True)
                
                # Update frame number spinbox
                self.frame_number_spinbox.setRange(0, total_frames - 1)
                self.frame_number_spinbox.setValue(0)
                
                # Clean up
                cap.release()
                
            except Exception as e:
                QMessageBox.warning(self,
                                    "Error Loading Video",
                                    f"Error: {str(e)}")
            finally:
                QApplication.restoreOverrideCursor()

    def init_video_player(self, video_path, fps):
        """Initialize the video player thread"""
        # Create and start the video player thread
        self.player_thread = VideoPlayerThread(video_path, fps)
        self.player_thread.update_frame.connect(self.on_frame_update)
        self.player_thread.start()
        
        # Set initial state to paused
        self.is_playing = False
        self.player_thread.paused = True
        
        # Set file name as window title prefix
        file_name = os.path.basename(video_path)
        self.setWindowTitle(f"Import Frames - {file_name}")

    def stop_video_player(self):
        """Stop the video player thread if it's running"""
        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.stop()
            self.player_thread = None
            self.is_playing = False
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def update_range_slider(self):
        """Update the range slider after loading a video"""
        if not hasattr(self, 'total_frames') or not self.total_frames:
            return
            
        # Enable sliders
        self.range_start_slider.setEnabled(True)
        self.range_end_slider.setEnabled(True)
        
        # Set range for all sliders
        self.range_start_slider.setRange(0, self.total_frames - 1)
        self.range_end_slider.setRange(0, self.total_frames - 1)
        
        # Set initial values
        self.range_start_slider.setValue(0)
        self.range_end_slider.setValue(self.total_frames - 1)
        
        # Update spinboxes
        self.range_start_spinbox.setRange(0, self.total_frames - 1)
        self.range_end_spinbox.setRange(0, self.total_frames - 1)
        self.range_start_spinbox.setValue(0)
        self.range_end_spinbox.setValue(self.total_frames - 1)
        
        # Update time range label
        duration = self.total_frames / self.fps
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.time_range_label.setText(
            f"Video duration: {minutes}m {seconds}s ({self.total_frames} frames @ {self.fps:.2f} fps)"
        )

    def update_range_slider_label(self):
        """Update labels when range sliders change"""
        start_frame = self.range_start_slider.value()
        end_frame = self.range_end_slider.value()
        
        # Ensure end frame is not before start frame
        if end_frame < start_frame:
            self.range_end_slider.setValue(start_frame)
            end_frame = start_frame
        
        # Update spinboxes without triggering their signals
        self.range_start_spinbox.blockSignals(True)
        self.range_end_spinbox.blockSignals(True)
        self.range_start_spinbox.setValue(start_frame)
        self.range_end_spinbox.setValue(end_frame)
        self.range_start_spinbox.blockSignals(False)
        self.range_end_spinbox.blockSignals(False)

    def range_spinbox_changed(self):
        """Handle changes to range spinboxes"""
        start_frame = self.range_start_spinbox.value()
        end_frame = self.range_end_spinbox.value()
        
        # Ensure end frame is not before start frame
        if end_frame < start_frame:
            self.range_end_spinbox.setValue(start_frame)
            end_frame = start_frame
        
        # Update sliders without triggering their signals
        self.range_start_slider.blockSignals(True)
        self.range_end_slider.blockSignals(True)
        self.range_start_slider.setValue(start_frame)
        self.range_end_slider.setValue(end_frame)
        self.range_start_slider.blockSignals(False)
        self.range_end_slider.blockSignals(False)
        
        # Update calculated frames
        self.update_calculated_frames()

    def update_calculated_frames(self):
        """Calculate and display the frames that will be extracted"""
        if not hasattr(self, 'total_frames') or not self.total_frames:
            self.calculated_frames_edit.setText("Load a video to calculate frames")
            return
        
        try:
            frame_indices = []
            
            if self.current_tab == "range":
                # Extract frames from range
                start_frame = self.range_start_slider.value()
                end_frame = self.range_end_slider.value()
                step = self.every_n_frames_spinbox.value()
                frame_indices = list(range(start_frame, end_frame + 1, step))
                # Always include end_frame if not already present and in range
                if end_frame not in frame_indices and 0 <= end_frame < self.total_frames:
                    frame_indices.append(end_frame)
                    frame_indices = sorted(frame_indices)
            elif self.current_tab == "specific":
                # Extract specific frames
                frame_str = self.specific_frames_edit.text().strip()
                if frame_str:
                    parts = frame_str.split(',')
                    for part in parts:
                        part = part.strip()
                        if '-' in part:
                            # Handle range (e.g., "5-10")
                            range_parts = part.split('-')
                            if len(range_parts) == 2:
                                try:
                                    start = int(range_parts[0].strip())
                                    end = int(range_parts[1].strip())
                                    frame_indices.extend(range(start, end + 1))
                                except ValueError:
                                    pass
                        else:
                            # Handle single frame number
                            try:
                                frame_indices.append(int(part))
                            except ValueError:
                                pass
                                
                    # Remove duplicates and sort
                    frame_indices = sorted(list(set(frame_indices)))
                    
                    # Filter out-of-range frames
                    frame_indices = [idx for idx in frame_indices if 0 <= idx < self.total_frames]
            
            if frame_indices:
                # Show preview of calculated frames
                if len(frame_indices) <= 10:
                    # Show all frame numbers if there are <= 10
                    preview_text = ", ".join(str(idx) for idx in frame_indices)
                else:
                    # Show first 5, last 5, and count in between
                    first_five = ", ".join(str(idx) for idx in frame_indices[:5])
                    last_five = ", ".join(str(idx) for idx in frame_indices[-5:])
                    middle_count = len(frame_indices) - 10
                    preview_text = f"{first_five}, ... ({middle_count} more) ..., {last_five}"
                    
                self.calculated_frames_edit.setText(f"{len(frame_indices)} frames: {preview_text}")
            else:
                self.calculated_frames_edit.setText("No frames selected")
                
        except Exception as e:
            self.calculated_frames_edit.setText(f"Error: {str(e)}")

    def import_frames(self, import_after=False):
        """Extract frames from video and optionally import them"""
        video_file = self.video_file_edit.text()
        frame_prefix = self.frame_prefix_edit.text()
        frame_ext = self.frame_ext_combo.currentText()
        
        output_dir = self.output_dir_edit.text()
        output_dir = f"{output_dir}/{os.path.basename(video_file).split('.')[0]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate inputs
        if not video_file or not os.path.exists(video_file):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid video file.")
            return
            
        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid output directory.")
            return
            
        if not frame_prefix:
            # Use video filename as default prefix
            frame_prefix = os.path.splitext(os.path.basename(video_file))[0]
            self.frame_prefix_edit.setText(frame_prefix)
        
        # Determine frames to extract
        frame_indices = []
        
        if self.current_tab == "range":
            start_frame = self.range_start_slider.value()
            end_frame = self.range_end_slider.value()
            step = self.every_n_frames_spinbox.value()
            frame_indices = list(range(start_frame, end_frame + 1, step))
            # Always include end_frame if not already present and in range
            if end_frame not in frame_indices and 0 <= end_frame < self.total_frames:
                frame_indices.append(end_frame)
                frame_indices = sorted(frame_indices)
        else:  # specific frames
            frame_str = self.specific_frames_edit.text().strip()
            if frame_str:
                parts = frame_str.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        range_parts = part.split('-')
                        if len(range_parts) == 2:
                            try:
                                start = int(range_parts[0].strip())
                                end = int(range_parts[1].strip())
                                frame_indices.extend(range(start, end + 1))
                            except ValueError:
                                pass
                    else:
                        try:
                            frame_indices.append(int(part))
                        except ValueError:
                            pass
                            
        # Remove duplicates and sort
        frame_indices = sorted(list(set(frame_indices)))
        
        # Filter out-of-range frames
        frame_indices = [idx for idx in frame_indices if 0 <= idx < self.total_frames]
        
        if not frame_indices:
            QMessageBox.warning(self, "No Frames", "No valid frames selected for extraction.")
            return
        
        # Confirm extraction
        confirmation = QMessageBox.question(
            self,
            "Confirm Extraction",
            f"Extract {len(frame_indices)} frames from the video?\n\n"
            f"This will save files with naming the pattern: {frame_prefix}_[frame_number].{frame_ext}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if confirmation != QMessageBox.Yes:
            return
            
        # Create progress dialog and store as instance variable
        self.progress = ProgressBar(self.main_window.annotation_window, "Extracting Frames")
        self.progress.show()
        self.progress.start_progress(len(frame_indices))
        
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Create frame extractor thread
        self.extractor_thread = FrameExtractorThread(
            video_file, output_dir, frame_prefix, frame_ext, frame_indices,
        )
        self.extractor_thread.progress_updated.connect(self.progress.set_value)
        self.extractor_thread.extraction_completed.connect(
            lambda paths: self.extraction_completed(paths, import_after)
        )
        self.extractor_thread.extraction_error.connect(self.extraction_error)
        
        # Start extraction
        self.extractor_thread.start()
        # (Do not close progress bar or restore cursor here)

    def extraction_completed(self, frame_paths, import_after):
        """Handle completion of frame extraction"""
        # Finish and close progress bar, restore cursor
        if hasattr(self, 'progress') and self.progress:
            self.progress.finish_progress()
            self.progress.stop_progress()
            self.progress.close()
            self.progress = None
        QApplication.restoreOverrideCursor()
        
        QMessageBox.information(
            self,
            "Extraction Complete",
            f"Successfully extracted {len(frame_paths)} frames."
        )
        
        # Store frame paths
        self.frame_paths = frame_paths
        
        if import_after and frame_paths:
            # Close dialog and import frames
            self.accept()
            
            # Tell main window to import these frames using IO.ImportImages
            self.main_window.import_images._process_image_files(frame_paths)

    def extraction_error(self, error_msg):
        """Handle errors during frame extraction"""
        # Finish and close progress bar, restore cursor
        if hasattr(self, 'progress') and self.progress:
            self.progress.finish_progress()
            self.progress.stop_progress()
            self.progress.close()
            self.progress = None
        QApplication.restoreOverrideCursor()
        
        QMessageBox.critical(
            self,
            "Extraction Error",
            f"An error occurred during frame extraction:\n{error_msg}"
        )