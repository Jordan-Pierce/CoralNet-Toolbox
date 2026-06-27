import os
import math
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit,
                             QDialog, QApplication, QMessageBox, QGroupBox,
                             QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QSlider,
                             QStyle, QTabWidget, QWidget, QCheckBox)
from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_window_icon

from coralnet_toolbox.Rasters.VideoRaster import VideoRaster
from coralnet_toolbox.Annotations.QtPatchAnnotation import PatchAnnotation
from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation
from coralnet_toolbox.Annotations.QtPolygonAnnotation import PolygonAnnotation
from coralnet_toolbox.Annotations.QtMultiPolygonAnnotation import MultiPolygonAnnotation
from coralnet_toolbox.Annotations.QtMaskAnnotation import MaskAnnotation


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
# Main Extract Frames Dialog
# ----------------------------------------------------------------------------------------------------------------------


class ImportFrames(QDialog):
    def __init__(self, main_window, parent=None, video_raster=None):
        super().__init__(parent)
        self.main_window = main_window

        # When launched from a VideoRaster row, the dialog operates in
        # "video-raster mode": the video file is pre-filled and locked, the
        # Keyframes tab and Include-annotations option become available, and
        # extracted frames can inherit the raster's per-frame annotations.
        self.video_raster = video_raster

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Extract Frames from Video")
        self.resize(500, 600)

        self.video_file = ""
        self.output_dir = ""
        self.total_frames = 0
        self.fps = 30.0

        self.frame_paths = []

        # Captured during import_frames so extraction_completed can rebuild the
        # per-frame output paths for annotation cloning.
        self.frame_prefix = ""
        self.frame_ext = "jpg"
        self.last_frame_indices = []

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

        # If a VideoRaster was supplied, pre-load it now that the UI exists.
        if self.video_raster is not None:
            self._load_video_raster()

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

        # Keyframes / Annotated-Frames tabs only make sense when a VideoRaster was
        # supplied (i.e. launched from the ImageWindow context menu). When launched
        # from the MainWindow menu there is no source raster, so they are omitted
        # entirely rather than shown disabled (Qt's disabled-tab styling is too
        # subtle to read as unavailable). Indices default to -1 so the
        # current_tab dispatch never matches a missing tab.
        self.keyframes_tab_index = -1
        self.annotated_tab_index = -1
        if self.video_raster is not None:
            # --- Keyframes Tab ---
            keyframes_tab = QWidget()
            keyframes_layout = QFormLayout()
            self.keyframes_label = QLabel("No keyframes available.")
            self.keyframes_label.setWordWrap(True)
            keyframes_layout.addRow("Starred Keyframes:", self.keyframes_label)
            keyframes_tab.setLayout(keyframes_layout)
            self.keyframes_tab_index = self.tab_widget.addTab(keyframes_tab, "Keyframes")

            # --- Annotated Frames Tab ---
            annotated_tab = QWidget()
            annotated_layout = QFormLayout()
            self.annotated_label = QLabel("No annotated frames available.")
            self.annotated_label.setWordWrap(True)
            annotated_layout.addRow("Frames with Annotations:", self.annotated_label)
            annotated_tab.setLayout(annotated_layout)
            self.annotated_tab_index = self.tab_widget.addTab(annotated_tab, "Annotated Frames")

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
        if index == 0:
            self.current_tab = "range"
        elif index == 1:
            self.current_tab = "specific"
        elif index == self.keyframes_tab_index:
            self.current_tab = "keyframes"
        elif index == self.annotated_tab_index:
            self.current_tab = "annotated"
        else:
            self.current_tab = "range"
        self.update_calculated_frames()

    def create_buttons_layout(self):
        buttons_layout = QVBoxLayout()

        # Include-annotations option. When checked, annotations on the source
        # video frames are re-created on the newly imported still images during
        # "Extract and Import". Only available in video-raster mode (launched from
        # the ImageWindow context menu); when launched from the MainWindow menu
        # there is no source raster, so the option is disabled and greyed out.
        self.include_annotations_checkbox = QCheckBox(
            "Include annotations (re-create on imported frames)"
        )
        in_video_mode = self.video_raster is not None
        self.include_annotations_checkbox.setChecked(in_video_mode)
        self.include_annotations_checkbox.setEnabled(in_video_mode)
        buttons_layout.addWidget(self.include_annotations_checkbox)

        action_buttons_layout = QHBoxLayout()

        self.extract_button = QPushButton("Extract")
        self.extract_import_button = QPushButton("Extract and Import")
        self.cancel_button = QPushButton("Cancel")

        self.extract_button.clicked.connect(lambda: self.import_frames(import_after=False))
        self.extract_import_button.clicked.connect(lambda: self.import_frames(import_after=True))
        self.cancel_button.clicked.connect(self.reject)

        action_buttons_layout.addWidget(self.extract_button)
        action_buttons_layout.addWidget(self.extract_import_button)
        action_buttons_layout.addWidget(self.cancel_button)

        buttons_layout.addLayout(action_buttons_layout)

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

    def _load_video_raster(self):
        """Pre-fill the dialog from a supplied VideoRaster (video-raster mode).

        The video file field is populated and locked, frame/fps metadata is read
        directly from the raster (no extra cv2 open), and the Keyframes summary
        is filled in.
        """
        raster = self.video_raster
        self.video_file = raster.image_path
        self.video_file_edit.setText(raster.image_path)
        self.video_file_edit.setEnabled(False)
        self.video_file_button.setEnabled(False)

        self.total_frames = int(raster.frame_count)
        self.fps = raster.fps or 30.0

        # Default the prefix to the video stem so output names are sensible.
        if not self.frame_prefix_edit.text().strip():
            self.frame_prefix_edit.setText(os.path.splitext(os.path.basename(raster.image_path))[0])

        self._update_keyframes_label()
        self._update_annotated_label()
        self.update_range_slider()
        self.update_calculated_frames()

    def _frame_idx_from_key(self, key):
        """Extract the integer frame index from a ``...::frame_N`` virtual path."""
        try:
            return int(key.split('::frame_', 1)[1])
        except (ValueError, IndexError):
            return None

    def _get_annotated_frame_indices(self):
        """Return a sorted list of frame indices that have annotations on the video.

        Covers both vector annotations (keyed under ``<video_path>::frame_<idx>``
        in image_annotations_dict) and per-frame semantic mask overlays (stored in
        the annotation window's batch_results_cache).
        """
        if self.video_raster is None:
            return []

        annotation_window = self.main_window.annotation_window
        prefix = self.video_raster.image_path + '::frame_'
        indices = set()

        # Vector annotations
        for key, annotations in annotation_window.image_annotations_dict.items():
            if key.startswith(prefix) and annotations:
                idx = self._frame_idx_from_key(key)
                if idx is not None:
                    indices.add(idx)

        # Per-frame semantic mask overlays
        cache = getattr(annotation_window, 'batch_results_cache', None) or {}
        for key, cached in cache.items():
            if not (isinstance(key, str) and key.startswith(prefix) and cached):
                continue
            mask_arr = cached.get('mask_arr') if isinstance(cached, dict) else None
            if mask_arr is not None and np.any(mask_arr):
                idx = self._frame_idx_from_key(key)
                if idx is not None:
                    indices.add(idx)

        return sorted(indices)

    def _update_annotated_label(self):
        """Refresh the Annotated Frames tab summary."""
        if self.video_raster is None:
            self.annotated_label.setText("No annotated frames available.")
            return

        annotated = self._get_annotated_frame_indices()
        if not annotated:
            self.annotated_label.setText("No annotated frames for this video.")
            return

        n = len(annotated)
        if n <= 12:
            preview = ", ".join(str(i) for i in annotated)
        else:
            first = ", ".join(str(i) for i in annotated[:6])
            last = ", ".join(str(i) for i in annotated[-6:])
            preview = f"{first}, ... ({n - 12} more) ..., {last}"
        self.annotated_label.setText(f"{n} annotated frames: {preview}")

    def _update_keyframes_label(self):
        """Refresh the Keyframes tab summary from the raster's starred frames."""
        if self.video_raster is None:
            self.keyframes_label.setText("No keyframes available.")
            return

        keyframes = sorted(self.video_raster.get_keyframes())
        if not keyframes:
            self.keyframes_label.setText("No keyframes starred for this video.")
            return

        n = len(keyframes)
        if n <= 12:
            preview = ", ".join(str(i) for i in keyframes)
        else:
            first = ", ".join(str(i) for i in keyframes[:6])
            last = ", ".join(str(i) for i in keyframes[-6:])
            preview = f"{first}, ... ({n - 12} more) ..., {last}"
        self.keyframes_label.setText(f"{n} keyframes: {preview}")

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

        elif self.current_tab == "keyframes":
            if self.video_raster is not None:
                indices = list(self.video_raster.get_keyframes())

        elif self.current_tab == "annotated":
            indices = self._get_annotated_frame_indices()

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

        # Capture extraction parameters so extraction_completed can rebuild the
        # per-frame output paths (needed for annotation cloning).
        self.output_dir = output_dir
        self.frame_prefix = frame_prefix
        self.frame_ext = frame_ext
        self.last_frame_indices = frame_indices

        # Skip frames whose output file already exists on disk — re-extracting
        # them is wasted work. They are still imported/cloned below (those steps
        # are idempotent and keyed on the full target path set).
        all_target_paths = [
            os.path.join(output_dir, f"{frame_prefix}_{idx}.{frame_ext}")
            for idx in frame_indices
        ]
        self.all_target_paths = all_target_paths
        indices_to_extract = [
            idx for idx, path in zip(frame_indices, all_target_paths)
            if not os.path.exists(path)
        ]
        skipped_existing = len(frame_indices) - len(indices_to_extract)

        # Confirm
        confirm_msg = f"Extract {len(indices_to_extract)} frames from the video?\n\n"
        if skipped_existing:
            confirm_msg += f"({skipped_existing} already exist on disk and will be reused.)\n\n"
        confirm_msg += f"Files will be named: {frame_prefix}_[frame_number].{frame_ext}"
        reply = QMessageBox.question(
            self, "Confirm Extraction",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # All frames already exist — skip extraction entirely and go straight to
        # import / annotation cloning. Pass an empty "newly extracted" list so the
        # summary reflects that everything was reused from disk.
        if not indices_to_extract:
            self.extraction_completed([], import_after)
            return

        # Show progress and start worker thread
        self.progress = ProgressBar(self.main_window.annotation_window, "Extracting Frames")
        self.progress.show()
        self.progress.start_progress(len(indices_to_extract))

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.extractor_thread = FrameExtractorThread(
            video_file, output_dir, frame_prefix, frame_ext, indices_to_extract
        )
        self.extractor_thread.progress_updated.connect(self.progress.set_value)
        self.extractor_thread.extraction_completed.connect(
            lambda paths: self.extraction_completed(paths, import_after)
        )
        self.extractor_thread.extraction_error.connect(self.on_extraction_error)
        self.extractor_thread.start()

    def extraction_completed(self, newly_extracted_paths, import_after):
        self._close_progress()
        QApplication.restoreOverrideCursor()

        # Import the full target set (newly extracted + pre-existing on disk);
        # _process_image_files is idempotent and skips paths already imported.
        import_paths = getattr(self, 'all_target_paths', None) or newly_extracted_paths
        self.frame_paths = import_paths

        cloned_count = 0
        if import_after and import_paths:
            self.accept()
            self.main_window.import_images._process_image_files(import_paths)

            # In video-raster mode, optionally re-create the source video-frame
            # annotations on the newly imported still images.
            if (self.video_raster is not None and
                    self.include_annotations_checkbox.isChecked()):
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    cloned_count = self._clone_annotations_to_frames(self.last_frame_indices)
                finally:
                    QApplication.restoreOverrideCursor()

        # Single summary dialog covering both extraction and annotation cloning.
        reused = len(import_paths) - len(newly_extracted_paths)
        message = f"Extracted {len(newly_extracted_paths)} frames"
        if reused > 0:
            message += f" ({reused} reused from disk)"
        message += "."
        if cloned_count:
            message += f"\nRe-created {cloned_count} annotation(s) on the imported frames."
        QMessageBox.information(self, "Extraction Complete", message)

    def _clone_annotations_to_frames(self, frame_indices):
        """Re-create source video-frame annotations on the imported still frames.

        Handles both vector annotations (keyed as ``video.mp4::frame_N`` in
        image_annotations_dict) and per-frame semantic mask overlays (stored in
        the annotation window's batch_results_cache). Each is cloned onto the
        corresponding extracted image (``<output_dir>/<prefix>_<idx>.<ext>``).
        """
        annotation_window = self.main_window.annotation_window
        label_window = self.main_window.label_window
        raster_manager = self.main_window.image_window.raster_manager
        video_path = self.video_raster.image_path
        cache = getattr(annotation_window, 'batch_results_cache', None) or {}

        type_map = {
            'PatchAnnotation': PatchAnnotation,
            'RectangleAnnotation': RectangleAnnotation,
            'PolygonAnnotation': PolygonAnnotation,
            'MultiPolygonAnnotation': MultiPolygonAnnotation,
        }

        # Per-image progress feedback.
        progress = ProgressBar(self.main_window.annotation_window, "Cloning Annotations")
        progress.show()
        progress.start_progress(len(frame_indices))

        cloned_count = 0
        try:
            for idx in frame_indices:
                src_path = VideoRaster.make_frame_path(video_path, idx)
                target_path = os.path.join(self.output_dir, f"{self.frame_prefix}_{idx}.{self.frame_ext}")

                # Skip if the target image was not actually imported.
                if target_path not in raster_manager.image_paths:
                    progress.update_progress()
                    continue

                # --- Vector annotations ---
                for annotation in annotation_window.image_annotations_dict.get(src_path, []):
                    try:
                        # Dispatch on the live object's class: to_dict() does not emit
                        # a 'type' field, so the class name is the reliable key.
                        cls = type_map.get(type(annotation).__name__)
                        if cls is None:
                            continue
                        data = annotation.to_dict()
                        data['image_path'] = target_path
                        data['id'] = str(uuid.uuid4())  # fresh id; do not collide with source
                        new_annotation = cls.from_dict(data, label_window)
                        annotation_window.add_annotation(new_annotation, record_action=False)
                        cloned_count += 1
                    except Exception as e:
                        print(f"[ExtractFrames] Failed to clone annotation on frame {idx}: {e}")

                # --- Per-frame semantic mask overlay ---
                # Masks are NOT registered via add_annotation (that would put them
                # in image_annotations_dict and make them selectable like vector
                # annotations, which crashes create_cropped_image). Instead attach
                # to the raster the same way project-load does, then refresh counts.
                cached = cache.get(src_path)
                mask_arr = cached.get('mask_arr') if isinstance(cached, dict) else None
                if mask_arr is not None and np.any(mask_arr):
                    try:
                        new_mask = self._build_mask_annotation(mask_arr, target_path, label_window)
                        if new_mask is not None:
                            target_raster = raster_manager.get_raster(target_path)
                            if target_raster is not None:
                                target_raster.mask_annotation = new_mask
                            annotation_window.annotation_manager.register_mask_annotation(new_mask)
                            self.main_window.image_window.update_image_annotations(
                                target_path, update_counts=False
                            )
                            cloned_count += 1
                    except Exception as e:
                        print(f"[ExtractFrames] Failed to clone mask on frame {idx}: {e}")

                progress.update_progress()
        finally:
            progress.finish_progress()
            progress.stop_progress()
            progress.close()

        return cloned_count

    def _build_mask_annotation(self, mask_arr, target_path, label_window):
        """Build a MaskAnnotation for ``target_path`` from a cached class-id array.

        The cached video-frame mask stores raw class ids but not the class->label
        mapping. That mapping is canonical on the source video raster's live
        MaskAnnotation (shared across all frames), so it is copied here to
        translate class ids back to project labels.
        """
        all_labels = list(label_window.labels)
        if not all_labels:
            return None

        new_mask = MaskAnnotation(
            image_path=target_path,
            mask_data=np.ascontiguousarray(mask_arr.astype(np.uint8)),
            initial_labels=all_labels,
        )

        # Copy the canonical class-id -> label map from the source raster's mask.
        src_mask = getattr(self.video_raster, 'mask_annotation', None)
        if src_mask is not None and getattr(src_mask, 'class_id_to_label_map', None):
            new_mask.class_id_to_label_map.clear()
            new_mask.label_id_to_class_id_map.clear()
            max_id = 0
            for class_id, label in src_mask.class_id_to_label_map.items():
                new_mask.class_id_to_label_map[class_id] = label
                new_mask.label_id_to_class_id_map[label.id] = class_id
                max_id = max(max_id, class_id)
            new_mask.next_class_id = max_id + 1
            new_mask.visible_label_ids = set(new_mask.label_id_to_class_id_map.keys())
            new_mask.invalidate_color_map()

        return new_mask

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