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

    `progress` carries **a number of annotations resolved**, never a percentage.
    It used to carry both: one step per crop on the success path, but a running
    *percentage* on each of the three skip paths. The receiver added whatever
    arrived to its own counter, so a batch where most images could not be opened
    reported percentages like 4288%, growing without bound. One unit, everywhere.

    `failed_ids` lists the annotations this run could not crop, so the caller can
    stop asking for them. Without it a single unreachable image is an infinite
    loop: the gallery re-runs its refresh when cropping finishes, finds the same
    annotations still uncropped, and starts another worker.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, annotations, raster_manager, parent=None):
        super().__init__(parent)
        self.annotations = list(annotations)
        self.raster_manager = raster_manager
        self._cancelled = False
        # Annotation ids this run could not produce a crop for. Read by the
        # caller once `finished` has been emitted.
        self.failed_ids = []
        # One example per reason, for a console line that names the actual
        # problem rather than leaving a silent no-op.
        self.failure_reasons = {}

    def cancel(self):
        self._cancelled = True

    def _give_up(self, anns, reason, image_path):
        """Record a group as un-croppable and report it as that many steps."""
        for ann in anns:
            self.failed_ids.append(getattr(ann, 'id', None))
        self.failure_reasons.setdefault(reason, image_path)
        self.progress.emit(len(anns))

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

            for image_path, anns in anns_by_image.items():
                if self._cancelled:
                    break

                raster = None
                try:
                    raster = self.raster_manager.get_raster(image_path)
                except Exception:
                    raster = None

                if not raster:
                    # No raster under this exact path. Almost always a path the
                    # project registered in a different form than the annotation
                    # carries, which is invisible from the gallery.
                    self._give_up(anns, "no raster registered under that path", image_path)
                    continue

                # Ensure rasterio source is available in this thread
                try:
                    if not hasattr(raster, '_rasterio_src') or raster._rasterio_src is None:
                        raster.load_rasterio()
                except Exception:
                    self._give_up(anns, "the image could not be opened", image_path)
                    continue

                rasterio_src = getattr(raster, '_rasterio_src', None)
                if rasterio_src is None:
                    self._give_up(anns, "the image opened but has no readable source",
                                  image_path)
                    continue

                for ann in anns:
                    if self._cancelled:
                        break
                    try:
                        ann.create_cropped_image(rasterio_src)
                    except Exception:
                        pass

                    if getattr(ann, 'cropped_image', None) is None:
                        self.failed_ids.append(getattr(ann, 'id', None))
                        self.failure_reasons.setdefault("the crop itself failed", image_path)

                    # One annotation resolved, cropped or not.
                    self.progress.emit(1)

            for reason, example in self.failure_reasons.items():
                print(f"Warning: could not crop annotations because {reason}: {example}")

            self.finished.emit()
        except Exception as e:
            try:
                self.error.emit(str(e))
            except Exception:
                pass
