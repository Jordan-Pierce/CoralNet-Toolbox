import os
import gc
import time
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QApplication, QMessageBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
                             QStyle, QFrame, QTabWidget, QWidget)
from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon, get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Video Player Thread
# ----------------------------------------------------------------------------------------------------------------------


class VideoPlayerThread(QThread):
    """Thread for playing video frames without blocking the UI."""
    update_frame = pyqtSignal(object, int)

    def __init__(self, video_path, fps=30):
        super().__init__()
        self.video_path = video_path
        self.fps = max(fps, 1)
        self.running = False
        self.paused = True
        self.mutex = QMutex()
        self.current_frame_number = 0
        self.total_frames = 0
        self._seek_requested = False
        self._seek_target = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0
        last_read_frame = -1  # tracks what frame cap is positioned at after last read

        while self.running:
            self.mutex.lock()
            paused = self.paused
            seek_requested = self._seek_requested
            seek_target = self._seek_target
            if seek_requested:
                self._seek_requested = False
            self.mutex.unlock()

            if seek_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                ret, frame = cap.read()
                if ret and frame is not None:
                    last_read_frame = seek_target
                    self.mutex.lock()
                    self.current_frame_number = seek_target
                    self.mutex.unlock()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.update_frame.emit(frame_rgb, seek_target)
                else:
                    last_read_frame = seek_target
                time.sleep(0.01)
                continue

            if paused:
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue

            self.mutex.lock()
            current = self.current_frame_number
            self.mutex.unlock()

            # Seek only if we're not already positioned at the right frame
            if last_read_frame != current - 1 and current != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current)

            ret, frame = cap.read()
            if not ret or frame is None:
                # End of video — wrap around
                self.mutex.lock()
                self.current_frame_number = 0
                self.mutex.unlock()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_read_frame = -1
                continue

            last_read_frame = current
            next_frame = (current + 1) % self.total_frames

            self.mutex.lock()
            self.current_frame_number = next_frame
            self.mutex.unlock()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.update_frame.emit(frame_rgb, current)
            last_frame_time = now

        cap.release()

    def seek(self, frame_number):
        """Seek to a specific frame (thread-safe)."""
        self.mutex.lock()
        self.current_frame_number = min(max(0, frame_number), max(0, self.total_frames - 1))
        self._seek_requested = True
        self._seek_target = self.current_frame_number
        self.mutex.unlock()

    def toggle_pause(self):
        """Toggle pause state (thread-safe)."""
        self.mutex.lock()
        self.paused = not self.paused
        self.mutex.unlock()

    def stop(self):
        """Stop the thread and wait for it to finish."""
        self.running = False
        self.wait(3000)


# ----------------------------------------------------------------------------------------------------------------------
# Frame Extraction Worker
# ----------------------------------------------------------------------------------------------------------------------


def _extract_chunk(video_file, frame_indices, output_dir, frame_prefix, ext):
    """
    Worker function for parallel extraction of a sorted chunk of frame indices.
    Opens its own VideoCapture, reads sequentially where possible.
    Returns list of successfully written paths.
    """
    if not frame_indices:
        return []

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    write_params = _get_write_params(ext)
    extracted = []
    last_pos = -2  # sentinel: no previous read

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= total_frames:
            continue
        # Only seek if not already at the right position
        if frame_idx != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        last_pos = frame_idx
        if not ret or frame is None:
            continue
        output_path = os.path.join(output_dir, f"{frame_prefix}_{frame_idx}.{ext}")
        success = cv2.imwrite(output_path, frame, write_params)
        if success:
            extracted.append(output_path)

    cap.release()
    return extracted


def _get_write_params(ext):
    """Return cv2.imwrite params for a given extension."""
    ext_lower = ext.lower()
    if ext_lower in ('jpg', 'jpeg'):
        return [cv2.IMWRITE_JPEG_QUALITY, 95]
    if ext_lower == 'png':
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 9 is max but very slow; 3 is a good balance
    return []


class FrameExtractorThread(QThread):
    """Thread for parallel frame extraction without blocking the UI."""
    progress_updated = pyqtSignal(int)
    extraction_completed = pyqtSignal(list)
    extraction_error = pyqtSignal(str)

    # Number of parallel workers. Each opens its own VideoCapture.
    NUM_WORKERS = 4

    def __init__(self, video_file, output_dir, frame_prefix, ext, frame_indices):
        super().__init__()
        self.video_file = video_file
        self.output_dir = output_dir
        self.frame_prefix = frame_prefix
        self.ext = ext
        # Always sort so each chunk is a contiguous sequential run
        self.frame_indices = sorted(set(frame_indices))

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            total = len(self.frame_indices)
            if total == 0:
                self.extraction_completed.emit([])
                return

            # Split into N sorted chunks — each worker reads its own sequential segment
            chunks = self._split_chunks(self.frame_indices, self.NUM_WORKERS)
            extracted_paths = []
            completed_count = 0

            with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _extract_chunk,
                        self.video_file,
                        chunk,
                        self.output_dir,
                        self.frame_prefix,
                        self.ext,
                    ): chunk
                    for chunk in chunks if chunk
                }

                for future in as_completed(futures):
                    try:
                        paths = future.result()
                        extracted_paths.extend(paths)
                        completed_count += len(futures[future])
                        self.progress_updated.emit(min(completed_count, total))
                    except Exception as e:
                        # Log but continue with other chunks
                        print(f"[FrameExtractor] Chunk error: {e}")

            # Re-sort output paths by frame index for deterministic ordering
            extracted_paths.sort(key=lambda p: self._frame_idx_from_path(p))
            self.extraction_completed.emit(extracted_paths)

        except Exception as e:
            self.extraction_error.emit(str(e))

    @staticmethod
    def _split_chunks(indices, n):
        """Split a sorted list into n roughly equal sorted chunks."""
        size = math.ceil(len(indices) / n) if n > 0 else len(indices)
        return [indices[i:i + size] for i in range(0, len(indices), size)]

    def _frame_idx_from_path(self, path):
        """Extract numeric frame index from output filename for sorting."""
        try:
            stem = os.path.splitext(os.path.basename(path))[0]
            return int(stem.rsplit('_', 1)[-1])
        except (ValueError, IndexError):
            return 0


# ----------------------------------------------------------------------------------------------------------------------
# Frame Cache (O(1) LRU via OrderedDict)
# ----------------------------------------------------------------------------------------------------------------------


class FrameCache:
    """LRU cache for video frames using OrderedDict for O(1) operations."""

    def __init__(self, max_size=30):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, frame_idx):
        if frame_idx not in self.cache:
            return None
        self.cache.move_to_end(frame_idx)
        return self.cache[frame_idx]

    def put(self, frame_idx, frame):
        if frame_idx in self.cache:
            self.cache.move_to_end(frame_idx)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[frame_idx] = frame

    def clear(self):
        self.cache.clear()


# ----------------------------------------------------------------------------------------------------------------------
# Main Import Frames Dialog
# ----------------------------------------------------------------------------------------------------------------------


class ImportFrames(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Import Frames from Video")
        self.resize(1000, 600)

        self.video_file = ""
        self.output_dir = ""
        self.total_frames = 0
        self.fps = 30.0

        self.frame_paths = []
        self.frame_cache = FrameCache(max_size=30)

        # Video player state
        self.player_thread = None
        self.is_playing = False

        # Build UI
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.create_info_group())
        controls_layout.addWidget(self.create_import_group())
        controls_layout.addWidget(self.create_output_group())
        controls_layout.addWidget(self.create_sample_group())
        controls_layout.addLayout(self.create_buttons_layout())

        preview_layout = self.setup_preview_layout()

        main_layout.addLayout(controls_layout, stretch=60)
        main_layout.addLayout(preview_layout, stretch=40)
        self.setLayout(main_layout)

        self.current_frame_idx = 0

        # Periodic UI sync timer (only updates when playing)
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(200)

    # ------------------------------------------------------------------
    # UI Group Builders
    # ------------------------------------------------------------------

    def create_info_group(self):
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        info_label = QLabel("Choose a video file to extract frames from.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        group_box.setLayout(layout)
        return group_box

    def create_import_group(self):
        group_box = QGroupBox("Import")
        layout = QFormLayout()

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

        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("Browse...")
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addRow("Output Directory:", output_dir_layout)

        prefix_layout = QHBoxLayout()
        self.frame_prefix_edit = QLineEdit()
        self.frame_prefix_edit.setPlaceholderText("frame_prefix_(idx).ext")
        prefix_layout.addWidget(self.frame_prefix_edit)
        layout.addRow("Frame Prefix:", prefix_layout)

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

        self.tab_widget = QTabWidget()
        self.current_tab = "range"

        # --- Range Tab ---
        range_tab = QWidget()
        range_layout = QFormLayout()

        self.time_range_label = QLabel("Time Range: No video loaded")

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

        self.every_n_frames_spinbox = QSpinBox()
        self.every_n_frames_spinbox.setRange(1, 10_000_000)
        self.every_n_frames_spinbox.setValue(24)
        self.every_n_frames_spinbox.valueChanged.connect(self.update_calculated_frames)

        range_layout.addRow("Sample Every N Frames:", self.every_n_frames_spinbox)
        range_layout.addRow("Select Start Frame:", self.range_start_slider)
        range_layout.addRow("Select End Frame:", self.range_end_slider)
        range_layout.addRow("", range_input_layout)
        range_layout.addRow("", self.time_range_label)
        range_tab.setLayout(range_layout)

        # --- Specific Frames Tab ---
        specific_tab = QWidget()
        specific_layout = QFormLayout()

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

        self.tab_widget.addTab(range_tab, "Frame Range")
        self.tab_widget.addTab(specific_tab, "Specific Frames")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        layout.addWidget(self.tab_widget)

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
        preview_layout = QVBoxLayout()

        preview_group = QGroupBox("Video Preview")
        preview_inner_layout = QVBoxLayout()

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("QLabel { background-color: black; }")
        preview_inner_layout.addWidget(self.preview_label)

        vc_layout = QHBoxLayout()

        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)

        self.prev_frame_btn = QPushButton("←")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)

        self.next_frame_btn = QPushButton("→")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)

        self.frame_number_spinbox = QSpinBox()
        self.frame_number_spinbox.setFixedWidth(80)
        self.frame_number_spinbox.setAlignment(Qt.AlignCenter)
        self.frame_number_spinbox.setEnabled(False)

        self.goto_frame_btn = QPushButton()
        self.goto_frame_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.goto_frame_btn.clicked.connect(self.goto_frame)
        self.goto_frame_btn.setEnabled(False)

        vc_layout.addWidget(self.play_pause_btn)
        vc_layout.addWidget(self.prev_frame_btn)
        vc_layout.addWidget(self.next_frame_btn)
        vc_layout.addWidget(self.frame_number_spinbox)
        vc_layout.addWidget(self.goto_frame_btn)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        preview_inner_layout.addLayout(vc_layout)
        preview_inner_layout.addWidget(self.frame_slider)
        preview_group.setLayout(preview_inner_layout)
        preview_layout.addWidget(preview_group)

        info_group = QGroupBox("Frame Information")
        info_layout = QVBoxLayout()

        self.frame_counter = QLabel("Frame: 0 / 0")
        self.frame_counter.setAlignment(Qt.AlignCenter)
        self.frame_counter.setStyleSheet(app_theme.scale_qss("QLabel { font-size: 11pt; }"))

        self.frame_timestamp = QLabel("Time: 00:00")
        self.frame_timestamp.setAlignment(Qt.AlignCenter)
        self.frame_timestamp.setStyleSheet(app_theme.scale_qss("QLabel { font-size: 11pt; }"))

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        info_layout.addWidget(self.frame_counter)
        info_layout.addWidget(line)
        info_layout.addWidget(self.frame_timestamp)

        info_group.setLayout(info_layout)
        preview_layout.addWidget(info_group)

        return preview_layout

    # ------------------------------------------------------------------
    # Playback Controls
    # ------------------------------------------------------------------

    def goto_frame(self):
        if not self.player_thread:
            return
        frame_num = self.frame_number_spinbox.value()
        self.current_frame_idx = frame_num
        # Block slider to avoid recursive slider_changed → seek loop
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_num)
        self.frame_slider.blockSignals(False)
        self.player_thread.seek(frame_num)

    def toggle_playback(self):
        if not self.player_thread:
            return
        self.player_thread.toggle_pause()
        self.is_playing = not self.is_playing
        icon = QStyle.SP_MediaPause if self.is_playing else QStyle.SP_MediaPlay
        self.play_pause_btn.setIcon(self.style().standardIcon(icon))

    def update_ui_elements(self):
        """Sync slider / counter to current playback position."""
        if not (self.player_thread and self.player_thread.isRunning() and self.is_playing):
            return

        current_frame = self.player_thread.current_frame_number
        total = self.player_thread.total_frames

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(current_frame)
        self.frame_slider.blockSignals(False)

        self.frame_counter.setText(f"Frame: {current_frame} / {total}")

        fps = self.player_thread.fps
        if fps > 0:
            t = current_frame / fps
            self.frame_timestamp.setText(f"Time: {int(t // 60):02d}:{int(t % 60):02d}")

    def on_frame_update(self, frame, frame_number):
        """Render a frame emitted from the player thread."""
        if frame is None:
            return

        h, w, _ = frame.shape
        q_img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)

        label_size = self.preview_label.size()
        img_ratio = w / h
        label_ratio = label_size.width() / label_size.height()

        if img_ratio > label_ratio:
            scaled_size = QSize(label_size.width(), int(label_size.width() / img_ratio))
        else:
            scaled_size = QSize(int(label_size.height() * img_ratio), label_size.height())

        pixmap = QPixmap.fromImage(q_img)
        self.preview_label.setPixmap(
            pixmap.scaled(scaled_size.width(), scaled_size.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        self.current_frame_idx = frame_number

        self.frame_number_spinbox.blockSignals(True)
        self.frame_number_spinbox.setValue(frame_number)
        self.frame_number_spinbox.blockSignals(False)

        self.frame_counter.setText(f"Frame: {frame_number} / {self.total_frames}")
        if self.fps > 0:
            t = frame_number / self.fps
            self.frame_timestamp.setText(f"Time: {int(t // 60):02d}:{int(t % 60):02d}")

    def slider_changed(self, value):
        self.current_frame_idx = value
        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.seek(value)
        self.frame_number_spinbox.blockSignals(True)
        self.frame_number_spinbox.setValue(value)
        self.frame_number_spinbox.blockSignals(False)

    def next_frame(self):
        if self.player_thread and self.player_thread.isRunning():
            target = min(self.current_frame_idx + 1, self.player_thread.total_frames - 1)
            self.player_thread.seek(target)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(target)
            self.frame_slider.blockSignals(False)

    def prev_frame(self):
        if self.player_thread and self.player_thread.isRunning():
            target = max(self.current_frame_idx - 1, 0)
            self.player_thread.seek(target)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(target)
            self.frame_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # File / Directory Browsers
    # ------------------------------------------------------------------

    def browse_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if not file_name:
            return

        if not os.path.exists(file_name):
            QMessageBox.warning(self, "Invalid Video File", "Please select a valid video file.")
            return

        self.video_file = file_name
        self.video_file_edit.setText(file_name)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.stop_video_player()

        try:
            cap = cv2.VideoCapture(file_name)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error Loading Video", "Could not open video file.")
                return

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            self._setup_controls_for_video()
            self.init_video_player(file_name, self.fps)
            self.update_range_slider()
            self.update_calculated_frames()

        except Exception as e:
            QMessageBox.warning(self, "Error Loading Video", f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _setup_controls_for_video(self):
        """Enable and configure all playback controls after a video is loaded."""
        max_frame = max(0, self.total_frames - 1)

        for widget in (self.play_pause_btn, self.prev_frame_btn,
                       self.next_frame_btn, self.frame_number_spinbox, self.goto_frame_btn):
            widget.setEnabled(True)

        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, max_frame)
        self.frame_slider.setValue(0)

        self.frame_number_spinbox.setRange(0, max_frame)
        self.frame_number_spinbox.setValue(0)

    def init_video_player(self, video_path, fps):
        self.player_thread = VideoPlayerThread(video_path, fps)
        self.player_thread.update_frame.connect(self.on_frame_update)
        self.player_thread.start()
        self.is_playing = False
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.setWindowTitle(f"Import Frames - {os.path.basename(video_path)}")
        # Show frame 0 immediately
        self.player_thread.seek(0)

    def stop_video_player(self):
        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.stop()
        self.player_thread = None
        self.is_playing = False
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    # ------------------------------------------------------------------
    # Range / Frame Selection
    # ------------------------------------------------------------------

    def update_range_slider(self):
        if not self.total_frames:
            return

        max_frame = self.total_frames - 1

        self.range_start_slider.setEnabled(True)
        self.range_end_slider.setEnabled(True)

        for slider, val in ((self.range_start_slider, 0), (self.range_end_slider, max_frame)):
            slider.setRange(0, max_frame)
            slider.setValue(val)

        for spinbox, val in ((self.range_start_spinbox, 0), (self.range_end_spinbox, max_frame)):
            spinbox.setRange(0, max_frame)
            spinbox.setValue(val)

        duration = self.total_frames / self.fps
        m, s = int(duration // 60), int(duration % 60)
        self.time_range_label.setText(
            f"Video duration: {m}m {s}s ({self.total_frames} frames @ {self.fps:.2f} fps)"
        )

    def update_range_slider_label(self):
        start = self.range_start_slider.value()
        end = self.range_end_slider.value()

        if end < start:
            self.range_end_slider.setValue(start)
            end = start

        self.range_start_spinbox.blockSignals(True)
        self.range_end_spinbox.blockSignals(True)
        self.range_start_spinbox.setValue(start)
        self.range_end_spinbox.setValue(end)
        self.range_start_spinbox.blockSignals(False)
        self.range_end_spinbox.blockSignals(False)

    def range_spinbox_changed(self):
        start = self.range_start_spinbox.value()
        end = self.range_end_spinbox.value()

        if end < start:
            self.range_end_spinbox.setValue(start)
            end = start

        self.range_start_slider.blockSignals(True)
        self.range_end_slider.blockSignals(True)
        self.range_start_slider.setValue(start)
        self.range_end_slider.setValue(end)
        self.range_start_slider.blockSignals(False)
        self.range_end_slider.blockSignals(False)

        self.update_calculated_frames()

    def _parse_frame_indices(self):
        """
        Derive the list of frame indices from the current UI state.
        Returns a sorted, deduplicated, in-range list.
        """
        if not self.total_frames:
            return []

        indices = []

        if self.current_tab == "range":
            start = self.range_start_slider.value()
            end = self.range_end_slider.value()
            step = self.every_n_frames_spinbox.value()
            indices = list(range(start, end + 1, step))
            # Ensure end frame is always included
            if indices and indices[-1] != end:
                indices.append(end)

        else:  # specific
            text = self.specific_frames_edit.text().strip()
            if text:
                for part in text.split(','):
                    part = part.strip()
                    if '-' in part:
                        bounds = part.split('-', 1)
                        try:
                            s, e = int(bounds[0].strip()), int(bounds[1].strip())
                            indices.extend(range(s, e + 1))
                        except ValueError:
                            pass
                    else:
                        try:
                            indices.append(int(part))
                        except ValueError:
                            pass

        # Deduplicate, sort, clamp
        return sorted({idx for idx in indices if 0 <= idx < self.total_frames})

    def update_calculated_frames(self):
        if not self.total_frames:
            self.calculated_frames_edit.setText("Load a video to calculate frames")
            return

        try:
            indices = self._parse_frame_indices()
            n = len(indices)
            if n == 0:
                self.calculated_frames_edit.setText("No frames selected")
                return

            if n <= 10:
                preview = ", ".join(str(i) for i in indices)
            else:
                first = ", ".join(str(i) for i in indices[:5])
                last = ", ".join(str(i) for i in indices[-5:])
                preview = f"{first}, ... ({n - 10} more) ..., {last}"

            self.calculated_frames_edit.setText(f"{n} frames: {preview}")

        except Exception as e:
            self.calculated_frames_edit.setText(f"Error: {e}")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def import_frames(self, import_after=False):
        """Validate inputs, confirm, then kick off background extraction."""

        # --- Validate inputs BEFORE touching the filesystem ---
        video_file = self.video_file_edit.text().strip()
        if not video_file or not os.path.exists(video_file):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid video file.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Invalid Input", "Please select an output directory.")
            return

        frame_prefix = self.frame_prefix_edit.text().strip()
        if not frame_prefix:
            frame_prefix = os.path.splitext(os.path.basename(video_file))[0]
            self.frame_prefix_edit.setText(frame_prefix)

        frame_ext = self.frame_ext_combo.currentText().strip().lstrip('.')
        frame_indices = self._parse_frame_indices()

        if not frame_indices:
            QMessageBox.warning(self, "No Frames", "No valid frames selected for extraction.")
            return

        # Build output subdirectory (after validation)
        video_stem = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(output_dir, video_stem)

        # Confirm
        reply = QMessageBox.question(
            self, "Confirm Extraction",
            f"Extract {len(frame_indices)} frames from the video?\n\n"
            f"Files will be named: {frame_prefix}_[frame_number].{frame_ext}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Show progress and start worker thread
        self.progress = ProgressBar(self.main_window.annotation_window, "Extracting Frames")
        self.progress.show()
        self.progress.start_progress(len(frame_indices))

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.extractor_thread = FrameExtractorThread(
            video_file, output_dir, frame_prefix, frame_ext, frame_indices
        )
        self.extractor_thread.progress_updated.connect(self.progress.set_value)
        self.extractor_thread.extraction_completed.connect(
            lambda paths: self.extraction_completed(paths, import_after)
        )
        self.extractor_thread.extraction_error.connect(self.on_extraction_error)
        self.extractor_thread.start()

    def extraction_completed(self, frame_paths, import_after):
        self._close_progress()
        QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self, "Extraction Complete",
            f"Successfully extracted {len(frame_paths)} frames."
        )

        self.frame_paths = frame_paths

        if import_after and frame_paths:
            self.accept()
            self.main_window.import_images._process_image_files(frame_paths)

    def on_extraction_error(self, error_msg):
        self._close_progress()
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(
            self, "Extraction Error",
            f"An error occurred during frame extraction:\n{error_msg}"
        )

    def _close_progress(self):
        if hasattr(self, 'progress') and self.progress:
            self.progress.finish_progress()
            self.progress.stop_progress()
            self.progress.close()
            self.progress = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Ensure background threads are stopped before the dialog closes."""
        self.ui_update_timer.stop()
        self.stop_video_player()
        if hasattr(self, 'extractor_thread') and self.extractor_thread and self.extractor_thread.isRunning():
            self.extractor_thread.wait(5000)
        super().closeEvent(event)