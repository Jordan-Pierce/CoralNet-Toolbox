import warnings
import os
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject

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

    def read(self, bands=None, window=None):
        """
        Return image data matching rasterio's read() interface.

        Args:
            bands: int (single band, 1-indexed) or list of ints, or None (all bands).
            window: rasterio Window-like object with col_off, row_off, width, height.

        Returns:
            ndarray: (H, W) for a single band, (B, H, W) for multiple bands.
        """
        # Resolve crop region from window
        if window is not None:
            col = max(0, min(int(window.col_off), self.width))
            row = max(0, min(int(window.row_off), self.height))
            w = max(1, min(int(window.width), self.width - col))
            h = max(1, min(int(window.height), self.height - row))
        else:
            col, row, w, h = 0, 0, self.width, self.height

        if self._current_bgr is None:
            # Blank frame
            if isinstance(bands, int):
                return np.zeros((h, w), dtype=np.uint8)
            n = len(bands) if isinstance(bands, list) else 3
            return np.zeros((n, h, w), dtype=np.uint8)

        # Crop and convert BGR → RGB, produce (3, H, W)
        rgb = self._current_bgr[:, :, ::-1]
        cropped = rgb[row:row + h, col:col + w]       # (H, W, 3)
        band_data = np.ascontiguousarray(np.transpose(cropped, (2, 0, 1)))  # (3, H, W)

        if isinstance(bands, int):
            return band_data[bands - 1]               # (H, W), 1-indexed
        elif isinstance(bands, list):
            indices = [b - 1 for b in bands]          # 1-indexed → 0-indexed
            return band_data[indices]                  # (len(bands), H, W)
        else:
            return band_data                           # (3, H, W)

    # Close is a no-op — required so Raster.cleanup() doesn't crash
    def close(self):
        pass


class VideoRaster(Raster):
    """
    A Raster subclass backed by a video file instead of a static image.

    Frames are addressed via virtual paths of the form:
        /path/to/video.mp4::frame_42

    The shim satisfies the rasterio interface so all annotation tools,
    cropping helpers, and the confidence window work without modification.
    """

    def __init__(self, video_path: str):
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
        data['type'] = 'VideoRaster'
        data['fps'] = self._video_fps
        data['frame_count'] = self._video_frame_count
        return data

    @classmethod
    def from_dict(cls, raster_dict: dict) -> 'VideoRaster':
        """Reconstruct a VideoRaster from a saved dictionary."""
        video_path = raster_dict['path']
        raster = cls(video_path)
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

    def cleanup(self):
        """Release cv2 resources before parent cleanup."""
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
