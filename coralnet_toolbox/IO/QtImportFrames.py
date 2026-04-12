import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QApplication, QMessageBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
                             QStyle, QTabWidget, QWidget)
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon


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
# Main Import Frames Dialog
# ----------------------------------------------------------------------------------------------------------------------


class ImportFrames(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Import Frames from Video")
        self.resize(1000, 600)

        self.video_file = ""
        self.output_dir = ""
        self.total_frames = 0
        self.fps = 30.0

        self.frame_paths = []

        # Build UI (controls only)
        main_layout = QVBoxLayout()
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.create_info_group())
        controls_layout.addWidget(self.create_import_group())
        controls_layout.addWidget(self.create_output_group())
        controls_layout.addWidget(self.create_sample_group())
        controls_layout.addLayout(self.create_buttons_layout())

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

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

        try:
            cap = cv2.VideoCapture(file_name)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error Loading Video", "Could not open video file.")
                return

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            # No preview: just update range controls
            self.update_range_slider()
            self.update_calculated_frames()

        except Exception as e:
            QMessageBox.warning(self, "Error Loading Video", f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    # Playback control helpers removed (preview disabled)

    # Video player initialization and stop removed (preview disabled)

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
        if hasattr(self, 'extractor_thread') and self.extractor_thread and self.extractor_thread.isRunning():
            self.extractor_thread.wait(5000)
        super().closeEvent(event)