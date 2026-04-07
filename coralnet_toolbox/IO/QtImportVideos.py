import os
import warnings

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Video extensions accepted for drag/drop and dialogs
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


class ImportVideos(QObject):
    """
    Handles importing one or more video files as first-class VideoRasters in the project.
    The video appears as a single row in the ImageWindow. Per-frame annotations
    are stored under virtual paths of the form 'video.mp4::frame_N'.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window

    def import_videos(self):
        """Open a file dialog and import one or more selected video files."""
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV *.MKV *.WEBM)"
        video_paths, _ = QFileDialog.getOpenFileNames(
            self.main_window,
            "Import Videos",
            "",
            file_filter,
        )
        if not video_paths:
            return

        # Make cursor busy while importing multiple files
        QApplication.setOverrideCursor(Qt.WaitCursor)
        failed = []
        added = []
        try:
            for video_path in video_paths:
                try:
                    success = self.main_window.image_window.raster_manager.add_video_raster(video_path)
                except Exception as e:
                    success = False
                    print(f"ImportVideos error for {video_path}: {e}")

                if success:
                    added.append(video_path)
                else:
                    failed.append(video_path)
        finally:
            QApplication.restoreOverrideCursor()

        if not added:
            QMessageBox.warning(
                self.main_window,
                "Import Failed",
                "Could not open any selected video files.\n\n"
                "Ensure the files are valid videos and OpenCV has the required codec.",
            )
            return

        if failed:
            QMessageBox.warning(
                self.main_window,
                "Partial Import",
                "Some files could not be imported:\n" + "\n".join(failed),
            )

        # Refresh the ImageWindow list once
        try:
            self.main_window.image_window.update_search_bars()
            self.main_window.image_window.filter_images(use_threading=False)
        except Exception:
            pass

    def dragEnterEvent(self, event):
        """Accept drag if it contains local video files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]
            lower_names = [fn.lower() for fn in file_names]
            if any(fn.endswith(tuple(VIDEO_EXTENSIONS)) for fn in lower_names):
                event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Allow moving drag over target for video files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]
            lower_names = [fn.lower() for fn in file_names]
            if any(fn.endswith(tuple(VIDEO_EXTENSIONS)) for fn in lower_names):
                event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        """Handle dropped files and import any supported videos."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]
            # Filter to supported video extensions
            video_files = [f for f in file_names if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
            if video_files:
                self._process_video_files(video_files)

    def _process_video_files(self, file_names):
        """Process a list of video files with a progress dialog."""
        QApplication.setOverrideCursor(Qt.WaitCursor)

        progress_bar = ProgressBar(self.main_window.image_window, title="Importing Videos")
        progress_bar.show()
        progress_bar.start_progress(len(file_names))

        try:
            imported_paths = []
            progress_batch = 0
            PROGRESS_BATCH_SIZE = 5

            for file_name in file_names:
                if file_name not in self.main_window.image_window.raster_manager.image_paths:
                    try:
                        if self.main_window.image_window.raster_manager.add_video_raster(file_name):
                            imported_paths.append(file_name)
                    except Exception as e:
                        print(f"ImportVideos error for {file_name}: {e}")
                else:
                    imported_paths.append(file_name)

                progress_batch += 1
                if progress_batch >= PROGRESS_BATCH_SIZE:
                    for _ in range(progress_batch):
                        progress_bar.update_progress()
                    progress_batch = 0

            # Flush remaining progress
            for _ in range(progress_batch):
                progress_bar.update_progress()

            # After the import loop, refresh UI once
            try:
                self.main_window.image_window.update_search_bars()
                self.main_window.image_window.filter_images(use_threading=False)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.warning(self.main_window,
                                "Error Importing Video(s)",
                                f"An error occurred while importing video(s): {e}")
        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.stop_progress()
            progress_bar.close()
