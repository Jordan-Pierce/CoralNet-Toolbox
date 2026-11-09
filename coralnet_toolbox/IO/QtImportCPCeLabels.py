import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.IO.QtImportCPCeAnnotations import read_cpce_code_file

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.utilities import STATUS_SUMMARY_MS, plural, show_status


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------
# Note: The labels come from a CPCe code file (.txt), the same file the CPCe annotation import optionally reads.


class ImportCPCeLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window

    def import_cpce_labels(self, file_path=None):
        self.main_window.untoggle_all_tools()

        if file_path is None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(self.label_window,
                                                       "Import CPCe Labels",
                                                       "",
                                                       "CPCe Code Files (*.txt);;All Files (*)",
                                                       options=options)
        if not file_path:
            return

        try:
            labels = read_cpce_code_file(file_path)

        except Exception as e:
            show_status(self.main_window, f"CPCe label import failed: {e}", STATUS_SUMMARY_MS)
            QMessageBox.warning(self.label_window,
                                "Error Loading Labels",
                                f"An error occurred while loading the CPCe code file: {str(e)}")
            return

        show_status(self.main_window, f"Importing {plural(len(labels), 'CPCe label')}...")

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self.label_window, "Importing CPCe Labels")
        progress_bar.show()
        progress_bar.start_progress(len(labels))

        added = 0
        error = None

        try:
            # Import the labels
            for short_label_code, long_label_code in labels:
                # Look up by short code alone, as the CPCe annotation import does, so a label it made
                # without the code file (long name == short code) is not duplicated here.
                if self.label_window.get_label_by_short_code(short_label_code, return_review=False) is None:
                    # Defer UI refresh; batch-refresh once after the loop to avoid flashing.
                    self.label_window.add_label_if_not_exists(short_label_code,
                                                              long_label_code,
                                                              refresh_ui=False)
                    added += 1

                # Update the progress bar
                progress_bar.update_progress()

        except Exception as e:
            error = e

        finally:
            # Single UI refresh for the whole batch (avoids per-label flashing).
            self.label_window.refresh_after_batch_add()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()
            QApplication.restoreOverrideCursor()

        # Reported after the progress bar is down, so the message box is reachable
        if error is not None:
            show_status(self.main_window, f"CPCe label import failed: {error}", STATUS_SUMMARY_MS)
            QMessageBox.warning(self.label_window,
                                "Error Importing Labels",
                                f"An error occurred while importing CPCe labels: {str(error)}")
            return

        message = f"Imported {plural(added, 'CPCe label')} from {os.path.basename(file_path)}."
        existing = len(labels) - added
        if existing:
            verb = "was" if existing == 1 else "were"
            message += f" {plural(existing, 'label')} already in the project {verb} left unchanged."
        show_status(self.main_window, message, STATUS_SUMMARY_MS)

        QMessageBox.information(self.label_window, "Labels Imported", message)
