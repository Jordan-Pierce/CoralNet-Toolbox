import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

import cv2

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QApplication, QMessageBox, QCheckBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
                             QStyle, QFrame, QTabWidget, QWidget)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportFrames(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window

        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Import Frames from Video")
        self.resize(800, 600)

        self.video_file = ""
        self.output_dir = ""
        self.num_frames = 0
        self.start_frame = 0
        self.end_frame = 0

        self.frame_paths = []

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
        """Set up the video preview panel"""
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

        # Frame navigation
        nav_layout = QHBoxLayout()

        self.prev_frame_btn = QPushButton("←")
        self.prev_frame_btn.clicked.connect(self.prev_frame)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        self.next_frame_btn = QPushButton("→")
        self.next_frame_btn.clicked.connect(self.next_frame)

        # Add frame number spinbox and reload button
        self.frame_number_spinbox = QSpinBox()
        self.frame_number_spinbox.setAlignment(Qt.AlignCenter)
        self.frame_number_spinbox.setFixedWidth(80)
        self.frame_number_spinbox.valueChanged.connect(self.frame_number_changed)

        self.frame_number_reload = QPushButton()
        self.frame_number_reload.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.frame_number_reload.clicked.connect(self.frame_number_changed)

        nav_layout.addWidget(self.prev_frame_btn)
        nav_layout.addWidget(self.next_frame_btn)
        nav_layout.addWidget(self.frame_number_spinbox)
        nav_layout.addWidget(self.frame_number_reload)

        preview_inner_layout.addLayout(nav_layout)
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

    def frame_number_changed(self):
        """Handle manual frame number input"""
        if self.cap is None:
            return

        frame_num = self.frame_number_spinbox.value()
        self.current_frame_idx = frame_num
        self.frame_slider.setValue(frame_num)

    def update_preview(self, frame_idx):
        """Update the preview with the specified frame"""
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            # Update frame number spinbox
            self.frame_number_spinbox.setValue(frame_idx)

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Scale frame to fit preview area while maintaining aspect ratio
            preview_size = self.preview_label.size()
            h, w = rgb_frame.shape[:2]
            aspect = w / h

            if preview_size.width() / preview_size.height() > aspect:
                new_h = preview_size.height()
                new_w = int(new_h * aspect)
            else:
                new_w = preview_size.width()
                new_h = int(new_w / aspect)

            scaled_frame = cv2.resize(rgb_frame, (new_w, new_h))

            # Convert to QImage and display
            h, w = scaled_frame.shape[:2]
            q_img = QImage(scaled_frame.data, w, h, scaled_frame.strides[0], QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(q_img))

            # Update counter and timestamp
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            time_in_seconds = frame_idx / fps
            minutes = int(time_in_seconds // 60)
            seconds = int(time_in_seconds % 60)

            self.frame_counter.setText(f"Frame: {frame_idx} / {total_frames}")
            self.frame_timestamp.setText(f"Time: {minutes:02d}:{seconds:02d}")

    def slider_changed(self, value):
        """Handle frame slider value changes"""
        self.current_frame_idx = value
        self.update_preview(value)

    def next_frame(self):
        """Show next frame"""
        if self.cap is not None:
            self.current_frame_idx = min(self.current_frame_idx + 1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
            self.frame_slider.setValue(self.current_frame_idx)

    def prev_frame(self):
        """Show previous frame"""
        if self.cap is not None:
            self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            self.frame_slider.setValue(self.current_frame_idx)

    def browse_video_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "Select Video File",
                                                   "",
                                                   "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
                                                   options=options)

        if file_name:
            if os.path.exists(file_name):
                self.video_file_edit.setText(file_name)

                # Close previous capture if exists
                if self.cap is not None:
                    self.cap.release()

                # Open new video capture
                self.cap = cv2.VideoCapture(file_name)
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Update frame slider
                self.frame_slider.setEnabled(True)
                self.frame_slider.setRange(0, total_frames - 1)
                self.frame_slider.setValue(0)
                self.current_frame_idx = 0

                # Set initial frame number in input
                self.frame_number_spinbox.setValue(0)

                # Update the range slider
                self.update_range_slider()

                # Show first frame
                self.update_preview(0)
            else:
                QMessageBox.warning(self, "Invalid Video File",
                                  "Please select a valid video file.")

    def browse_output_dir(self):
        options = QFileDialog.Options()
        dir_name = QFileDialog.getExistingDirectory(self,
                                                    "Select Output Directory",
                                                    options=options)

        if dir_name:
            if os.path.exists(dir_name):
                self.output_dir_edit.setText(dir_name)

    def update_range_slider_label(self):
        """Update the range spinboxes with current values."""
        start = self.range_start_slider.value()
        end = self.range_end_slider.value()
        self.range_start_spinbox.setValue(start)
        self.range_end_spinbox.setValue(end)
        self.update_time_label()

    def update_range_slider(self):
        """Update the range slider based on the selected video file."""
        if self.video_file_edit.text():
            try:
                cap = cv2.VideoCapture(self.video_file_edit.text())
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)

                # Enable the slider and set its range
                self.range_start_slider.setEnabled(True)
                self.range_end_slider.setEnabled(True)
                self.range_start_spinbox.setEnabled(True)
                self.range_end_spinbox.setEnabled(True)

                # Set ranges for spinboxes
                self.range_start_spinbox.setRange(0, total_frames)
                self.range_end_spinbox.setRange(0, total_frames)
                self.frame_number_spinbox.setRange(0, total_frames - 1)

                tick_interval = max(1, total_frames // 10)
                self.range_start_slider.setRange(0, total_frames)
                self.range_end_slider.setRange(0, total_frames)
                self.range_start_slider.setTickInterval(tick_interval)
                self.range_end_slider.setTickInterval(tick_interval)
                self.range_start_slider.setValue(0)
                self.range_end_slider.setValue(total_frames)

                # Update the spinbox values
                self.range_start_spinbox.setValue(0)
                self.range_end_spinbox.setValue(total_frames)

                self.update_time_label()
                self.update_calculated_frames()

                cap.release()

            except Exception as e:
                # Handle potential errors
                print(f"Error reading video file: {e}")
                self.range_start_slider.setValue(0)
                self.range_end_slider.setValue(0)
                self.range_start_slider.setEnabled(False)
                self.range_end_slider.setEnabled(False)
                self.range_start_spinbox.setEnabled(False)
                self.range_end_spinbox.setEnabled(False)
                self.range_start_spinbox.setValue(0)
                self.range_end_spinbox.setValue(0)
                self.time_range_label.setText("Time Range: Unable to read video file")
                self.calculated_frames_edit.setText("Invalid video file")

    def range_spinbox_changed(self):
        """Handle manual frame range spinbox changes"""
        if not self.range_start_slider.isEnabled() or not self.range_end_slider.isEnabled():
            return

        # Get current values
        start = self.range_start_spinbox.value()
        end = self.range_end_spinbox.value()

        # Ensure start doesn't exceed end
        if start > end:
            if self.sender() == self.range_start_spinbox:
                start = end
                self.range_start_spinbox.setValue(start)
            else:
                end = start
                self.range_end_spinbox.setValue(end)

        # Update the slider with new values
        self.range_start_slider.setValue(start)
        self.range_end_slider.setValue(end)

    def update_time_label(self):
        """Update the time range label based on fps and selected range."""
        try:
            start = self.range_start_slider.value()
            end = self.range_end_slider.value()
            start_time = start / self.fps
            end_time = end / self.fps

            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            end_min = int(end_time // 60)
            end_sec = int(end_time % 60)

            time_text = f"Time Range: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
            self.time_range_label.setText(time_text)
        except:
            self.time_range_label.setText("Time Range: Unable to calculate")

    def update_calculated_frames(self):
        """Calculate and display the number of frames that will be extracted."""
        try:
            frame_count = 0

            if self.current_tab == "range":
                if self.range_start_slider.isEnabled() and self.range_end_slider.isEnabled():
                    # Get range slider frames
                    start = self.range_start_slider.value()
                    end = self.range_end_slider.value()
                    every_n = self.every_n_frames_spinbox.value()
                    frame_count = len(range(start, end, every_n))
            else:  # specific frames tab
                if self.specific_frames_edit.text().strip():
                    frame_count = len(self.parse_specific_frames())

            self.calculated_frames_edit.setText(f"{frame_count} frames will be extracted")

        except Exception as e:
            self.calculated_frames_edit.setText("Unable to calculate frames")

    def parse_specific_frames(self):
        """
        Parse the frame ranges string into a list of frame numbers.

        :param frame_ranges_str:
        """
        frames = []
        # Get the specific frames as a list of integers
        ranges = self.specific_frames_edit.text().split(',')
        for r in ranges:
            r = r.strip()
            if not r:
                continue
            if '-' in r:
                start, end = map(int, r.split('-'))
                frames.extend(range(start, end + 1))
            else:
                frames.append(int(r))

        return sorted(set(frames))

    def import_frames(self, import_after=False):
        """Import frames from the video file."""
        # Get the video file
        self.video_file = self.video_file_edit.text()

        # Create a directory for the frames
        self.output_dir = f"{self.output_dir_edit.text()}/{os.path.basename(self.video_file).split('.')[0]}"
        self.output_dir = self.output_dir.replace("\\", "/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get the frame prefix
        self.frame_prefix = self.frame_prefix_edit.text()
        self.frame_prefix = "frame" if not self.frame_prefix else self.frame_prefix

        # Get the frame extension, and other values
        self.ext = self.frame_ext_combo.currentText().replace(".", "").lower()
        self.every_n_frames = self.every_n_frames_spinbox.value()
        self.start_frame = self.range_start_slider.value()
        self.end_frame = self.range_end_slider.value()

        if not self.video_file or not self.output_dir:
            QMessageBox.warning(self,
                                "Input Error",
                                "Please select a video file, output directory, and specify the number of frames.")
            return

        cap = cv2.VideoCapture(self.video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.start_frame >= total_frames or self.end_frame > total_frames or self.start_frame >= self.end_frame:
            QMessageBox.warning(self,
                                "Range Error",
                                "Invalid frame range selected.")
            return

        frame_indices = self.get_frame_indices(total_frames)
        self.save_frames(cap, frame_indices)
        cap.release()

        if import_after:
            self.import_images()

        self.accept()

    def get_frame_indices(self, total_frames):
        """Get the frame indices based on the active tab selection"""
        frame_indices = set()

        if self.current_tab == "range":
            # Use range slider and every_n_frames
            for i in range(self.start_frame, self.end_frame, self.every_n_frames_spinbox.value()):
                frame_indices.add(i)
        else:
            # Use specific frames
            specific_frames = self.parse_specific_frames()
            for f in specific_frames:
                if 0 <= f < total_frames:
                    frame_indices.add(f)

        return sorted(frame_indices)

    def save_frames(self, cap, frame_indices):
        """Save the frames to the output directory."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.image_window, title="Extracting Frames")
        progress_bar.show()
        progress_bar.start_progress(len(frame_indices))

        # Clear the frame paths
        self.frame_paths = []

        try:
            for idx, frame_index in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if not ret:
                    print(f"Failed to read frame {frame_index}")
                    continue

                frame_name = f"{self.output_dir}/{self.frame_prefix}_{frame_index}.{self.ext}"
                if not cv2.imwrite(frame_name, frame):
                    print(f"Failed to write frame to {frame_name}")
                    continue

                self.frame_paths.append(frame_name)
                progress_bar.update_progress()

            QMessageBox.information(self,
                                    "Success",
                                    f"Successfully extracted {len(self.frame_paths)} frames")

        except Exception as e:
            QMessageBox.critical(self,
                                 "Unexpected Error",
                                 f"An unexpected error occurred: {str(e)}")
            self.frame_paths = []

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def import_images(self):
        """Import the extracted frames into the application."""
        if not self.frame_paths:
            return
            
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            progress_bar = ProgressBar(self.annotation_window, title="Importing Frames")
            progress_bar.show()
            progress_bar.start_progress(len(self.frame_paths))
            
            # Import each frame
            for frame_path in self.frame_paths:
                # Add the image to the window
                self.image_window.add_image(frame_path)
                
                # Update the progress bar
                progress_bar.update_progress()
                
            # Load the first image
            if self.frame_paths:
                # The add_image call already triggers a refresh of the image display
                # so we don't need any additional logic here to select an image
                pass
                
        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Frames",
                                f"An error occurred while importing frames: {str(e)}")
                                
        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()

    def closeEvent(self, event):
        """Clean up resources when dialog is closed"""
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)
