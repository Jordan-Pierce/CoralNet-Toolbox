import warnings
import os
import time
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, QThread, QMutex, pyqtSignal

from coralnet_toolbox.Rasters.QtRaster import Raster

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM'}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class VideoRasterioShim:
    """
    Lightweight mock that satisfies the rasterio interface expected by annotation
    cropping methods (create_cropped_image).  No actual rasterio I/O is performed.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.count = 3  # RGB bands
        self.closed = False  # Required: rasterio_to_cropped_image checks getattr(src, 'closed', True)
        self._current_bgr: Optional[np.ndarray] = None  # (H, W, 3) BGR

    def read(self, bands=None, window=None, out_shape=None, resampling=None):
        # Resolve crop region
        if window is not None:
            col = max(0, min(int(window.col_off), self.width))
            row = max(0, min(int(window.row_off), self.height))
            w = max(1, min(int(window.width), self.width - col))
            h = max(1, min(int(window.height), self.height - row))
        else:
            col, row, w, h = 0, 0, self.width, self.height

        # Determine desired output size (out_shape may be (bands, H, W) or (H, W))
        out_h, out_w = h, w
        if out_shape is not None:
            try:
                if isinstance(out_shape, (tuple, list)):
                    if len(out_shape) == 3:
                        _, out_h, out_w = map(int, out_shape)
                    elif len(out_shape) == 2:
                        out_h, out_w = map(int, out_shape)
            except Exception:
                out_h, out_w = h, w

        # Empty frame -> return zeros with requested shape
        if self._current_bgr is None:
            if isinstance(bands, int):
                return np.zeros((out_h, out_w), dtype=np.uint8)
            n = len(bands) if isinstance(bands, list) else 3
            return np.zeros((n, out_h, out_w), dtype=np.uint8)

        # Crop and convert BGR -> RGB
        rgb = self._current_bgr[:, :, ::-1]
        cropped = rgb[row:row + h, col:col + w]  # (h, w, 3)

        # Resize if requested
        if (out_h != h) or (out_w != w):
            interp = cv2.INTER_LINEAR
            try:
                name = getattr(resampling, "name", None)
                if name == "nearest":
                    interp = cv2.INTER_NEAREST
                elif name == "bilinear":
                    interp = cv2.INTER_LINEAR
                elif name == "cubic":
                    interp = cv2.INTER_CUBIC
            except Exception:
                pass
            resized = cv2.resize(cropped, (int(out_w), int(out_h)), interpolation=interp)
        else:
            resized = cropped

        band_data = np.ascontiguousarray(np.transpose(resized, (2, 0, 1)))  # (3, H, W)

        if isinstance(bands, int):
            return band_data[bands - 1]
        elif isinstance(bands, list):
            indices = [b - 1 for b in bands]
            return band_data[indices]
        else:
            return band_data


class VideoDecodeWorker(QThread):
    """
    Decodes video frames on a background thread so the UI thread is never
    blocked by file I/O or pixel-format conversion.

    A single ``cv2.VideoCapture`` is kept open for the lifetime of the worker.
    During playback the worker self-paces with ``time.monotonic()`` rather
    than relying on QTimer, which eliminates timing jitter.

    Seek requests from the main thread are accepted via a mutex-protected
    slot and processed on the next loop iteration; the worker always emits
    one preview frame immediately after seeking so scrub-bar feedback is
    instant even when paused.
    """

    frameReady = pyqtSignal(int, object)   # frame_idx, QImage

    def __init__(self, video_path: str, fps: float,
                 start_frame: int = 0,
                 playback_max_width: int = 0,
                 parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._fps = max(fps, 1.0)
        self._start_frame = start_frame
        # 0 = no downscale; set to e.g. 1920 to cap playback resolution
        self._playback_max_width = playback_max_width

        self._running = False
        self._paused = True
        self._mutex = QMutex()
        self._seek_target: Optional[int] = None
        self._current_idx: int = start_frame
        self._total_frames: int = 0

        # Drop-frame gate: prevents frame signal queue buildup when the main
        # thread is slow.  Worker skips an emit if the previous one has not
        # been consumed yet.  Main-thread slot resets this flag after paint.
        self._pending_emit: bool = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        """Decode loop.  Runs entirely on the worker thread."""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 1.0 / self._fps
        self._running = True
        last_frame_time = 0.0

        if self._current_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_idx)

        while self._running:
            # ---- Collect protected state ----
            self._mutex.lock()
            seek = self._seek_target
            self._seek_target = None
            paused = self._paused
            self._mutex.unlock()

            # ---- Handle a seek request ----
            if seek is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                self._current_idx = seek
                ret, bgr = cap.read()
                if ret and bgr is not None:
                    self._pending_emit = False  # reset so the preview frame is never dropped
                    self._emit_frame(bgr, seek)
                    self._current_idx = seek + 1
                continue

            # ---- Sleep while paused ----
            if paused:
                time.sleep(0.005)
                continue

            # ---- Pace playback with wall clock (no QTimer drift) ----
            now = time.monotonic()
            sleep_for = frame_interval - (now - last_frame_time)
            if sleep_for > 0.001:
                time.sleep(sleep_for)

            # ---- Loop at end of video ----
            if self._current_idx >= self._total_frames:
                self._current_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ---- Sequential read — no seek overhead ----
            ret, bgr = cap.read()
            if not ret or bgr is None:
                self._current_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self._emit_frame(bgr, self._current_idx)
            self._current_idx += 1
            last_frame_time = time.monotonic()

        cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_frame(self, bgr: np.ndarray, frame_idx: int):
        """
        Convert a BGR frame to ``QImage`` and emit ``frameReady``.
        Skips the emit if the previous frame signal has not been consumed
        yet to avoid unbounded queue growth in the Qt event system.
        """
        if self._pending_emit:
            return  # Drop: main thread hasn't processed the previous frame yet

        self._pending_emit = True

        h, w = bgr.shape[:2]

        # Optional display-resolution downscale (helps smooth 4K playback)
        if self._playback_max_width > 0 and w > self._playback_max_width:
            scale = self._playback_max_width / w
            new_w = self._playback_max_width
            new_h = int(h * scale)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w

        # In-place BGR→RGB channel flip (no full numpy copy)
        rgb = bgr[..., ::-1]
        # .tobytes() copies the data so Qt does not hold a dangling buffer
        q_img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        self.frameReady.emit(frame_idx, q_img)

    # ------------------------------------------------------------------
    # Thread-safe API (callable from the main thread)
    # ------------------------------------------------------------------

    def seek(self, frame_idx: int):
        """Request a seek.  Thread-safe; can be called from any thread."""
        self._mutex.lock()
        if self._total_frames > 0:
            self._seek_target = max(0, min(frame_idx, self._total_frames - 1))
        else:
            self._seek_target = max(0, frame_idx)
        self._mutex.unlock()

    def set_paused(self, paused: bool):
        """Pause or resume the decode loop.  Thread-safe."""
        self._mutex.lock()
        self._paused = paused
        self._mutex.unlock()

    def stop(self):
        """Signal the decode loop to exit and wait (up to 3 s) for the thread."""
        self._running = False
        self.wait(3000)


class VideoRaster(Raster):
    """
    A Raster subclass backed by a video file instead of a static image.

    Frames are addressed via virtual paths of the form:
        /path/to/video.mp4::frame_42

    The shim satisfies the rasterio interface so all annotation tools,
    cropping helpers, and the confidence window work without modification.
    """

    # Emitted by the decode worker (forwarded via _on_worker_frame).
    # Listeners receive (frame_idx: int, q_img: QImage).
    frameReady = pyqtSignal(int, object)

    def __init__(self, video_path: str):
        # Canonical type
        self.raster_type = "VideoRaster"

        # Open the video capture before calling super().__init__ so that
        # load_rasterio() (called inside super().__init__) can use it.
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._video_frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._shim = VideoRasterioShim(self._video_width, self._video_height)
        self._current_frame_idx: int = 0

        # Thumbnail cache (populated on first call to get_thumbnail)
        self._video_thumbnail: Optional[QImage] = None

        # Background decode worker (created by start_decode_worker; None when stopped)
        self._decode_worker: Optional[VideoDecodeWorker] = None

        # Call parent __init__ — it will call load_rasterio() which we override
        super().__init__(video_path)

    # ------------------------------------------------------------------
    # Raster overrides
    # ------------------------------------------------------------------

    def load_rasterio(self) -> bool:
        """Override: populate dimensions from cv2, set shim as rasterio_src."""
        try:
            if not hasattr(self, '_shim') or self._shim is None:
                return False

            self.width = self._video_width
            self.height = self._video_height
            self.channels = 3
            self.shape = (self.height, self.width, self.channels)

            self.metadata['dimensions'] = f"{self.width}x{self.height}"
            self.metadata['bands'] = 3
            self.metadata['fps'] = self._video_fps
            self.metadata['frame_count'] = self._video_frame_count

            # Point rasterio_src at the shim
            self._rasterio_src = self._shim
            return True

        except Exception as e:
            print(f"VideoRaster.load_rasterio error for {self.image_path}: {e}")
            return False

    @property
    def rasterio_src(self):
        """Always return the shim so annotations can crop from the current frame."""
        return self._shim

    # ------------------------------------------------------------------
    # Video-specific properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._video_fps

    @property
    def frame_count(self) -> int:
        return self._video_frame_count

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_idx: int) -> Optional[QImage]:
        """
        Seek to frame_idx, read BGR, update the shim, and return a QImage.
        Returns None on failure.
        """
        if not self._cap or not self._cap.isOpened():
            return None

        frame_idx = max(0, min(frame_idx, self._video_frame_count - 1))

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        self._shim._current_bgr = bgr
        self._current_frame_idx = frame_idx

        return self._bgr_to_qimage(bgr)

    def get_bgr_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return a raw BGR numpy array for the given frame index."""
        if not self._cap or not self._cap.isOpened():
            return None

        frame_idx = max(0, min(frame_idx, self._video_frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        self._shim._current_bgr = bgr
        self._current_frame_idx = frame_idx
        return bgr

    def update_shim_for_frame(self, frame_idx: int):
        """Update the shim's current BGR data without returning a QImage."""
        if not self._cap or not self._cap.isOpened():
            return

        frame_idx = max(0, min(frame_idx, self._video_frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = self._cap.read()
        if ret and bgr is not None:
            self._shim._current_bgr = bgr
            self._current_frame_idx = frame_idx

    # ------------------------------------------------------------------
    # Decode worker lifecycle
    # ------------------------------------------------------------------

    def start_decode_worker(self, start_frame: int = 0) -> None:
        """
        Create and start the background :class:`VideoDecodeWorker` (paused by default).

        Safe to call multiple times — any existing worker is stopped first.
        Connect to :attr:`frameReady` *after* calling this method to receive frames.
        """
        self.stop_decode_worker()
        self._decode_worker = VideoDecodeWorker(
            self.image_path, self._video_fps, start_frame=start_frame
        )
        self._decode_worker.frameReady.connect(self._on_worker_frame)
        self._decode_worker.start()

    def stop_decode_worker(self) -> None:
        """Stop and destroy the background decode worker."""
        if self._decode_worker is not None:
            try:
                self._decode_worker.frameReady.disconnect()
            except Exception:
                pass
            self._decode_worker.stop()
            self._decode_worker = None

    def resume_decode_worker(self) -> None:
        """Unpause the decode worker (begin emitting frames)."""
        if self._decode_worker is not None:
            self._decode_worker._pending_emit = False  # Reset drop-gate before unpausing
            self._decode_worker.set_paused(False)

    def pause_decode_worker(self) -> None:
        """Pause the decode worker without stopping it."""
        if self._decode_worker is not None:
            self._decode_worker.set_paused(True)

    def seek_decode_worker(self, frame_idx: int) -> None:
        """
        Seek the decode worker to *frame_idx*.

        The worker will emit one preview frame immediately (even while paused),
        which is useful for scrub-bar feedback without a full annotation reload.
        """
        if self._decode_worker is not None:
            self._decode_worker.seek(frame_idx)

    def _on_worker_frame(self, frame_idx: int, q_img: QImage) -> None:
        """Internal slot: forward worker frames and keep ``_current_frame_idx`` in sync."""
        self._current_frame_idx = frame_idx
        self.frameReady.emit(frame_idx, q_img)

    # ------------------------------------------------------------------
    # Raster image access overrides
    # ------------------------------------------------------------------

    def get_qimage(self) -> Optional[QImage]:
        """Return frame 0 for display purposes (used by ImageWindow thumbnail)."""
        return self.get_frame(0)

    def get_thumbnail(self, longest_edge: int = 256) -> Optional[QImage]:
        """Return a cached thumbnail of frame 0."""
        if self._video_thumbnail is None:
            frame_image = self.get_frame(0)
            if frame_image is None:
                return None
            # Scale down
            pixmap = QPixmap.fromImage(frame_image)
            scaled = pixmap.scaled(
                longest_edge, longest_edge,
                aspectRatioMode=1,  # Qt.KeepAspectRatio
                transformMode=1     # Qt.SmoothTransformation
            )
            self._video_thumbnail = scaled.toImage()

        return self._video_thumbnail

    def get_pixmap(self, longest_edge: Optional[int] = None) -> Optional[QPixmap]:
        """Return a pixmap of frame 0."""
        q_image = self.get_qimage()
        if q_image is None:
            return None
        pixmap = QPixmap.fromImage(q_image)
        if longest_edge is not None:
            pixmap = pixmap.scaled(
                longest_edge, longest_edge,
                aspectRatioMode=1,
                transformMode=1
            )
        return pixmap

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Extend base serialization with video-specific fields."""
        data = super().to_dict()
        # Canonical raster type
        data['raster_type'] = 'VideoRaster'
        data['fps'] = self._video_fps
        data['frame_count'] = self._video_frame_count
        return data

    @classmethod
    def from_dict(cls, raster_dict: dict) -> 'VideoRaster':
        """Reconstruct a VideoRaster from a saved dictionary."""
        video_path = raster_dict['path']
        raster = cls(video_path)
        # Let the base class restore common properties (work areas, scale,
        # intrinsics, z-channel, etc.) while preserving the subclass's
        # `raster_type` (see Raster.update_from_dict for conditional logic).
        try:
            raster.update_from_dict(raster_dict)
        except Exception:
            # Fallback: at least restore simple state if update_from_dict fails
            state = raster_dict.get('state', {})
            raster.checkbox_state = state.get('checkbox_state', False)
        return raster

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_frame_path(video_path: str, frame_idx: int) -> str:
        """Create the virtual image path for a specific frame."""
        return f"{video_path}::frame_{frame_idx}"

    @staticmethod
    def parse_frame_path(path: str):
        """
        Parse a virtual frame path.
        Returns (video_path, frame_idx) or (path, None) if not a virtual path.
        """
        if '::frame_' in path:
            parts = path.rsplit('::frame_', 1)
            try:
                return parts[0], int(parts[1])
            except (ValueError, IndexError):
                pass
        return path, None

    @staticmethod
    def is_video_path(path: str) -> bool:
        """Return True if path has a recognised video extension."""
        ext = os.path.splitext(path)[1]
        return ext in VIDEO_EXTENSIONS

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, collect_garbage: bool = True):
        """Release cv2 resources before parent cleanup."""
        # Stop the background decode worker first so its cap is released before ours
        self.stop_decode_worker()

        if hasattr(self, '_cap') and self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        self._video_thumbnail = None

        # Clear shim reference (don't call close on parent's _rasterio_src
        # because the base class cleanup() will try to close it)
        self._rasterio_src = None
        if hasattr(self, '_shim'):
            self._shim = None

        # Skip Raster.cleanup()'s attempt to close rasterio_src
        # by calling QObject cleanup manually
        try:
            import gc
            self._q_image = None
            self._thumbnail = None
            self.annotations = []
            self.work_areas = []
            self.delete_mask_annotation()
            self.intrinsics = None
            self.extrinsics = None
            self.z_channel = None
            if collect_garbage:
                gc.collect()
        except Exception:
            pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
        """Convert a BGR numpy array to a QImage (RGB888)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
