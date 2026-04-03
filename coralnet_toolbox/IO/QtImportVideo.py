import warnings

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportVideo(QObject):
    """
    Handles importing a video file as a first-class VideoRaster in the project.
    The video appears as a single row in the ImageWindow.  Per-frame annotations
    are stored under virtual paths of the form 'video.mp4::frame_N'.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

    def import_video(self):
        """Open a file dialog and import the selected video file."""
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV *.MKV *.WEBM)"
        video_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Import Video",
            "",
            file_filter,
        )
        if not video_path:
            return

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            success = self.main_window.image_window.raster_manager.add_video_raster(video_path)
        except Exception as e:
            success = False
            print(f"ImportVideo error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

        if not success:
            QMessageBox.warning(
                self.main_window,
                "Import Failed",
                f"Could not open video file:\n{video_path}\n\n"
                "Ensure the file is a valid video and OpenCV has the required codec.",
            )
            return

        # Refresh the ImageWindow list
        try:
            self.main_window.image_window.update_search_bars()
            self.main_window.image_window.filter_images(use_threading=False)
        except Exception:
            pass
