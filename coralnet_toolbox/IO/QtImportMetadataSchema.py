import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QApplication)

from coralnet_toolbox.MetaData import MetaDataSchema

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportMetadataSchema:
    """Merge metadata field definitions from a YAML file into the project.

    Merging is by field name. An incoming definition never silently replaces an
    existing one, because the existing one may already govern stored values
    across the project -- the user is asked instead.
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window

    def import_metadata_schema(self):
        """Prompt for a YAML file and merge its fields into the project schema."""
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.annotation_window,
                                                   "Import Metadata Schema",
                                                   "",
                                                   "YAML Files (*.yaml *.yml);;All Files (*)",
                                                   options=options)
        if not file_path:
            return

        try:
            incoming = MetaDataSchema.from_yaml(file_path)
        except Exception as e:
            QMessageBox.warning(self.annotation_window,
                                "Error Loading Schema",
                                f"An error occurred while reading the metadata schema: {str(e)}")
            return

        if not len(incoming):
            QMessageBox.warning(self.annotation_window,
                                "Empty Schema",
                                "The selected file contains no metadata field definitions.")
            return

        metadata_window = self.main_window.metadata_window
        schema = metadata_window.schema

        # Ask once about collisions rather than field by field.
        collisions = [field.name for field in incoming if schema.has_field(field.name)]
        replace_existing = False

        if collisions:
            names = ", ".join(collisions[:10])
            if len(collisions) > 10:
                names += f", and {len(collisions) - 10} more"

            msg_box = QMessageBox(self.annotation_window)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Fields Already Exist")
            msg_box.setText(f"{len(collisions)} field(s) already exist in this project:\n\n{names}")
            msg_box.setInformativeText(
                "<b>Replace</b> overwrites the existing definitions. Stored values are kept, "
                "but any that do not fit the new definition will be discarded.<br><br>"
                "<b>Keep</b> leaves the existing definitions alone and imports only the new fields."
            )
            replace_button = msg_box.addButton("Replace", QMessageBox.DestructiveRole)
            keep_button = msg_box.addButton("Keep", QMessageBox.AcceptRole)
            msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(keep_button)
            msg_box.exec_()

            clicked = msg_box.clickedButton()
            if clicked is None or clicked not in (replace_button, keep_button):
                return
            replace_existing = clicked is replace_button

        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            added, skipped, replaced = schema.merge(incoming, replace_existing=replace_existing)

            # A replaced definition can invalidate values stored under the old
            # one, so bring them back in line rather than leaving them unreadable.
            discarded = 0
            if replaced:
                annotations = metadata_window.get_all_annotations()
                for name in replaced:
                    field = schema.get_field(name)
                    if field is not None:
                        _kept, dropped = schema.recoerce(annotations, field)
                        discarded += dropped

            metadata_window.set_schema(schema)

            message = f"Schema imported ({len(added)} new fields)."
            try:
                self.main_window.status_bar.showMessage(message, 3000)
            except Exception:
                pass

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self.annotation_window,
                                "Error Importing Schema",
                                f"An error occurred while importing the metadata schema: {str(e)}")
            return

        finally:
            # Restore the cursor
            QApplication.restoreOverrideCursor()

        summary = [f"<li>Added <b>{len(added)}</b> field(s).</li>"]
        if replaced:
            summary.append(f"<li>Replaced <b>{len(replaced)}</b> existing field(s).</li>")
        if discarded:
            summary.append(f"<li>Discarded <b>{discarded}</b> stored value(s) that no longer fit.</li>")
        if skipped:
            summary.append(f"<li>Kept <b>{len(skipped)}</b> existing field(s) unchanged.</li>")

        msg_box = QMessageBox(self.annotation_window)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Schema Imported")
        msg_box.setText("The metadata schema has been imported.")
        msg_box.setInformativeText(f"<ul>{''.join(summary)}</ul>")
        msg_box.exec_()
