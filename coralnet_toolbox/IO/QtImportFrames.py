import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

import cv2

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QDialog, QApplication, QMessageBox, QCheckBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider)

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
        self.resize(400, 300)

        self.video_file = ""
        self.output_dir = ""
        self.num_frames = 0
        self.start_frame = 0
        self.end_frame = 0
        
        self.frame_paths = []
        
        # Create the layout
        self.layout = QVBoxLayout(self)
        
        # Setup the info layout
        self.setup_info_layout()
        # Setup the import layout
        self.setup_import_layout()
        # Setup the output layout
        self.setup_output_layout()
        # Setup the sample layout
        self.setup_sample_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        """
        Set up the layout and widgets for the info layout.
        """
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()
        
        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Choose a video file to extract frames from.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_import_layout(self):
        """Set up the layout and widgets for the import layout."""
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
        self.layout.addWidget(group_box)

    def setup_output_layout(self):
        """Set up the layout and widgets for the output layout."""
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
        self.layout.addWidget(group_box)
        
    def setup_sample_layout(self):
        """Set up the layout and widgets for the sample layout."""
        group_box = QGroupBox("Sample Frames")
        layout = QFormLayout()
        
        # Every N frames to sample
        self.every_n_frames_spinbox = QSpinBox()
        self.every_n_frames_spinbox.setRange(1, 10000000)
        self.every_n_frames_spinbox.setValue(24)
        self.every_n_frames_spinbox.valueChanged.connect(self.update_calculated_frames)
        layout.addRow("Sample Every N Frames:", self.every_n_frames_spinbox)
        
        # Range slider for selecting frames
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(1, 1)  
        self.range_slider.setValue((1, 1))
        self.range_slider.setTickPosition(QSlider.TicksBelow)
        self.range_slider.setTickInterval(10)
        self.range_slider.valueChanged.connect(self.update_range_slider_label)
        self.range_slider.valueChanged.connect(self.update_calculated_frames)
        self.range_slider_label = QLabel("Select Frame Range: No video loaded")
        layout.addRow("Select Frame Range:", self.range_slider)
        layout.addRow("", self.range_slider_label)
        
        # Calculated frames display
        self.calculated_frames_edit = QLineEdit()
        self.calculated_frames_edit.setReadOnly(True)
        self.calculated_frames_edit.setText("Load a video to calculate frames")
        layout.addRow("Calculated Frames:", self.calculated_frames_edit)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
    
    def setup_buttons_layout(self):
        """Set up the layout and widgets for the buttons layout."""
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
        
        self.layout.addLayout(buttons_layout)

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
                self.update_range_slider()
            else:
                QMessageBox.warning(self,
                                    "Invalid Video File",
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
        """Update the range slider label with current values."""
        start, end = self.range_slider.value()
        self.range_slider_label.setText(f"{start} - {end}")

    def update_range_slider(self):
        """Update the range slider based on the selected video file."""
        if self.video_file_edit.text():
            try:
                cap = cv2.VideoCapture(self.video_file_edit.text())
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Enable the slider and set its range
                self.range_slider.setRange(0, total_frames)
                
                # Set tick interval to 10% of total frames
                tick_interval = max(1, total_frames // 10)
                self.range_slider.setTickInterval(tick_interval)
                # Set initial values to the full range
                self.range_slider.setValue((0, total_frames))
                # Update the label and calculated frames
                self.range_slider_label.setText(f"0 - {total_frames}")
                self.update_calculated_frames()
                
                cap.release()
                
            except Exception as e:
                # Handle potential errors in video file reading
                print(f"Error reading video file: {e}")
                self.range_slider.setRange(1, 1)
                self.range_slider.setValue((1, 1))
                self.range_slider_label.setText("Unable to read video file")
                self.calculated_frames_edit.setText("Invalid video file")
                
    def update_calculated_frames(self):
        """Calculate and display the number of frames that will be extracted."""
        try:
            start, end = self.range_slider.value()
            every_n = self.every_n_frames_spinbox.value()
            
            # Calculate sampled frames
            sampled_frames = len(range(start, end, every_n))
            self.calculated_frames_edit.setText(f"{sampled_frames} frames will be extracted")
            
        except Exception as e:
            self.calculated_frames_edit.setText("Unable to calculate frames")

    def import_frames(self, import_after=False):
        """Import frames from the video file."""
        # Get the video file
        self.video_file = self.video_file_edit.text()
        
        # Create a directory for the frames
        self.output_dir = f"{self.output_dir_edit.text()}/{os.path.basename(self.video_file).split('.')[0]}/"
        self.output_dir = self.output_dir.replace("\\", "/")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get the frame prefix
        self.frame_prefix = self.frame_prefix_edit.text()
        self.frame_prefix = "frame" if not self.frame_prefix else self.frame_prefix
        
        # Get the frame extension, and other values
        self.ext = self.frame_ext_combo.currentText().replace(".", "").lower()
        self.every_n_frames = self.every_n_frames_spinbox.value()
        self.start_frame, self.end_frame = self.range_slider.value()
        
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
        """Get the frame indices based on the start, end, and every_n_frames values."""
        frame_indices = []
        for i in range(self.start_frame, self.end_frame, self.every_n_frames):
            frame_indices.append(i)
        return frame_indices

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
        """Import the saved frames to the image window."""
        # Make the cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.image_window, title="Importing Images")
        progress_bar.show()
        progress_bar.start_progress(len(self.frame_paths))

        try:
            # Add images to the image window
            for idx, frame_path in enumerate(self.frame_paths):
                if frame_path not in set(self.image_window.image_paths):
                    try:
                        self.image_window.add_image(frame_path)
                    except Exception as e:
                        print(f"Warning: Could not import image {frame_path}. Error: {e}")

                # Update the progress bar
                progress_bar.update_progress()

            # Update filtered images
            self.image_window.filter_images()
            # Show the last image
            self.image_window.load_image_by_path(self.image_window.image_paths[-1])

            QMessageBox.information(self.image_window,
                                    "Frame(s) Imported",
                                    "Frame(s) have been successfully imported.")
        except Exception as e:
            QMessageBox.warning(self.image_window,
                                "Error Importing Frame(s)",
                                f"An error occurred while importing frame(s): {str(e)}")
        finally:
            # Restore the cursor to the default cursor
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
        