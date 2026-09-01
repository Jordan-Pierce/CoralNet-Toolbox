import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QApplication)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ExportMetadataSchema:
    """Write the project's metadata field definitions to a YAML file.

    Only the definitions travel, never the per-annotation values, so a schema
    can be shared between projects as a reusable template.
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def export_metadata_schema(self):
        """Prompt for a path and write the schema out."""
        self.main_window.untoggle_all_tools()

        schema = self.main_window.metadata_window.schema
        if not len(schema):
            QMessageBox.warning(self.annotation_window,
                                "No Metadata Fields",
                                "This project has no metadata fields to export.\n\n"
                                "Add fields in the Metadata panel first.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.annotation_window,
                                                   "Export Metadata Schema",
                                                   "",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)",
                                                   options=options)
        if not file_path:
            return

        if not file_path.lower().endswith(('.yaml', '.yml')):
            file_path += '.yaml'

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            schema.to_yaml(file_path)

            message = f"Schema exported ({len(schema)} fields)."
            QMessageBox.information(self.annotation_window,
                                    "Schema Exported",
                                    f"{len(schema)} metadata field(s) have been exported.")
            try:
                self.main_window.status_bar.showMessage(message, 3000)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Exporting Schema",
                                f"An error occurred while exporting the metadata schema: {str(e)}")

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()
