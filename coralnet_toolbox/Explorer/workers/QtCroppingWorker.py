"""
Cropping Worker for the Explorer.

Background QThread worker that generates cropped annotation images
so the gallery can display thumbnails without blocking the UI.
Extracted from ui/QtAnnotationViewerWindow.py.
"""

import warnings

from PyQt5.QtCore import QThread, pyqtSignal

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CroppingWorker(QThread):
    """Background worker to create cropped images for a set of annotations.

    Emits integer progress percentages (0-100), and `finished` when done.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, annotations, raster_manager, parent=None):
        super().__init__(parent)
        self.annotations = list(annotations)
        self.raster_manager = raster_manager
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # Group annotations by image_path
            anns_by_image = {}
            for ann in self.annotations:
                if not hasattr(ann, 'cropped_image') or ann.cropped_image is None:
                    anns_by_image.setdefault(ann.image_path, []).append(ann)

            total = sum(len(v) for v in anns_by_image.values())
            if total == 0:
                self.finished.emit()
                return

            processed = 0
            last_percent = -1
            for image_path, anns in anns_by_image.items():
                if self._cancelled:
                    break

                raster = None
                try:
                    raster = self.raster_manager.get_raster(image_path)
                except Exception:
                    raster = None

                if not raster:
                    # skip this image
                    processed += len(anns)
                    percent = int((processed / total) * 100)
                    if percent != last_percent:
                        last_percent = percent
                        self.progress.emit(percent)
                    continue

                # Ensure rasterio source is available in this thread
                try:
                    if not hasattr(raster, '_rasterio_src') or raster._rasterio_src is None:
                        raster.load_rasterio()
                except Exception:
                    # Can't open raster; skip
                    processed += len(anns)
                    percent = int((processed / total) * 100)
                    if percent != last_percent:
                        last_percent = percent
                        self.progress.emit(percent)
                    continue

                rasterio_src = getattr(raster, '_rasterio_src', None)
                if rasterio_src is None:
                    processed += len(anns)
                    percent = int((processed / total) * 100)
                    if percent != last_percent:
                        last_percent = percent
                        self.progress.emit(percent)
                    continue

                for ann in anns:
                    if self._cancelled:
                        break
                    try:
                        ann.create_cropped_image(rasterio_src)
                    except Exception:
                        # Non-fatal: continue with next
                        pass

                    processed += 1
                    # Emit a single-step progress event (worker will emit one
                    # signal per processed annotation). The caller will increment
                    # the ProgressBar via `update_progress()` which increments
                    # by one per call.
                    self.progress.emit(1)

            self.finished.emit()
        except Exception as e:
            try:
                self.error.emit(str(e))
            except Exception:
                pass
