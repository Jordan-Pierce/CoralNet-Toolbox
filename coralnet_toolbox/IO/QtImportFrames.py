import cv2
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit, QDialog, QApplication, QMessageBox, QCheckBox)
from qtrangeslider import QRangeSlider

class ImportFramesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Frames from Video")
        self.setGeometry(100, 100, 400, 300)

        self.video_file = ""
        self.output_dir = ""
        self.num_frames = 0
        self.start_frame = 0
        self.end_frame = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.video_file_label = QLabel("Video File: Not Selected")
        self.select_video_button = QPushButton("Select Video File")
        self.select_video_button.clicked.connect(self.select_video_file)

        self.output_dir_label = QLabel("Output Directory: Not Selected")
        self.select_output_dir_button = QPushButton("Select Output Directory")
        self.select_output_dir_button.clicked.connect(self.select_output_dir)

        self.num_frames_label = QLabel("Number of Frames to Sample:")
        self.num_frames_input = QLineEdit()

        self.range_slider_label = QLabel("Select Frame Range:")
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))

        self.import_checkbox = QCheckBox("Import images after extraction")
        self.import_checkbox.setChecked(True)

        self.import_button = QPushButton("Import Frames")
        self.import_button.clicked.connect(self.import_frames)

        layout.addWidget(self.video_file_label)
        layout.addWidget(self.select_video_button)
        layout.addWidget(self.output_dir_label)
        layout.addWidget(self.select_output_dir_button)
        layout.addWidget(self.num_frames_label)
        layout.addWidget(self.num_frames_input)
        layout.addWidget(self.range_slider_label)
        layout.addWidget(self.range_slider)
        layout.addWidget(self.import_checkbox)
        layout.addWidget(self.import_button)

        self.setLayout(layout)

    def select_video_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.video_file = file_name
            self.video_file_label.setText(f"Video File: {file_name}")
            self.update_range_slider()

    def select_output_dir(self):
        options = QFileDialog.Options()
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        if dir_name:
            self.output_dir = dir_name
            self.output_dir_label.setText(f"Output Directory: {dir_name}")

    def update_range_slider(self):
        if self.video_file:
            cap = cv2.VideoCapture(self.video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.range_slider.setRange(0, total_frames)
            self.range_slider.setValue((0, total_frames))
            cap.release()

    def import_frames(self):
        if not self.video_file or not self.output_dir or not self.num_frames_input.text():
            QMessageBox.warning(self, "Input Error", "Please select a video file, output directory, and specify the number of frames.")
            return

        self.num_frames = int(self.num_frames_input.text())
        self.start_frame, self.end_frame = self.range_slider.value()

        cap = cv2.VideoCapture(self.video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.start_frame >= total_frames or self.end_frame > total_frames or self.start_frame >= self.end_frame:
            QMessageBox.warning(self, "Range Error", "Invalid frame range selected.")
            return

        frame_indices = self.get_frame_indices(total_frames)
        self.save_frames(cap, frame_indices)
        cap.release()

        if self.import_checkbox.isChecked():
            self.import_images()

    def get_frame_indices(self, total_frames):
        frame_indices = []
        step = (self.end_frame - self.start_frame) // self.num_frames
        for i in range(self.num_frames):
            frame_indices.append(self.start_frame + i * step)
        return frame_indices

    def save_frames(self, cap, frame_indices):
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_name = f"{self.output_dir}/frame_{idx}.jpg"
                cv2.imwrite(frame_name, frame)
            else:
                QMessageBox.warning(self, "Frame Error", f"Failed to read frame at index {idx}.")
                return
        QMessageBox.information(self, "Success", "Frames have been successfully imported.")

    def import_images(self):
        # Placeholder for the import images logic
        pass
