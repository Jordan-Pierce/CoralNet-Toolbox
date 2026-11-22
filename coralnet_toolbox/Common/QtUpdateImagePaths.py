import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QHBoxLayout, QApplication, QAbstractItemView,
                             QMessageBox, QDialog, QListWidget, QPushButton, QFileDialog)

from coralnet_toolbox.Icons import get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def _path_parts(path):
    """Split a path into lower-cased components, whatever separator it was saved with.

    A project saved on Windows and opened elsewhere carries backslashes, which
    os.path treats as ordinary characters off Windows, so split by hand.
    """
    return [part for part in path.replace("\\", "/").lower().split("/") if part]


def _basename_key(path):
    """Case-insensitive basename, independent of the separator."""
    parts = _path_parts(path)
    return parts[-1] if parts else ""


def _shared_tail(a_parts, b_parts):
    """Number of trailing path components two paths have in common."""
    count = 0
    for a, b in zip(reversed(a_parts), reversed(b_parts)):
        if a != b:
            break
        count += 1
    return count


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class UpdateImagePaths(QDialog):
    """Relink project images whose files are no longer where the project says.

    Images can be found by searching a folder (matched by file name) or located
    one at a time (any file name). Whatever is still missing can be left behind
    with Continue; the caller opens the project with the images it has.
    """

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.setWindowIcon(get_window_icon("coral"))
        self.setWindowTitle("Update Image Paths")
        self.resize(520, 480)

        # Original Image Paths
        self.image_paths = list(image_paths)
        self.missing_images = self.find_missing_images()
        self.total_missing = len(self.missing_images)

        # Mapping between original and updated paths
        self.updated_paths = {}

        # Missing images the user chose to drop from the project for good
        self.removed_images = []

        # Folder the last manual locate came from, to start the next one there
        self.last_locate_dir = ""

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Setup the information layout
        self.setup_info_layout()
        # Setup the list layout
        self.setup_list_widget()
        # Setup the buttons layout
        self.setup_buttons_layout()

        self.refresh()

    def find_missing_images(self):
        """Returns a list of image paths that don't exist (each listed once)."""
        return list(dict.fromkeys(path for path in self.image_paths if not os.path.exists(path)))

    def setup_info_layout(self):
        """Create information label explaining missing images."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        hint_label = QLabel("<b>Search Folder</b>: look through a folder and all its subfolders for "
                            "files with the same name.<br><br>"
                            "<b>Locate</b>: pick the file for the selected image by hand, so a renamed "
                            "file can be found too.<br><br>"
                            "<b>Remove</b>: drop the selected images and their annotations from the "
                            "project.<br><br>"
                            "<b>Continue</b>: open the project without the images still missing. "
                            "They keep their annotations in the project file.")
        hint_label.setTextFormat(Qt.RichText)
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        group_box.setLayout(layout)
        self.layout().addWidget(group_box)

    def setup_list_widget(self):
        """Create list widget showing missing image paths."""
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setToolTip("Double click an image to locate its file.\n"
                                    "Shift or Ctrl click to select several to remove.")
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.locate_selected_image())
        self.list_widget.itemSelectionChanged.connect(self.update_buttons)
        self.layout().addWidget(self.list_widget)

    def setup_buttons_layout(self):
        """Create Search Folder, Locate and Continue buttons."""
        button_layout = QHBoxLayout()

        self.search_button = QPushButton("Search Folder...")
        self.search_button.clicked.connect(self.update_image_paths)
        self.search_button.setToolTip("Search a folder and its subfolders for the missing images.\n"
                                      "Images are matched by file name.")

        self.locate_button = QPushButton("Locate...")
        self.locate_button.clicked.connect(self.locate_selected_image)
        self.locate_button.setToolTip("Pick the file for the selected image.\nThe file name does not have to match.")

        self.remove_button = QPushButton("Remove...")
        self.remove_button.clicked.connect(self.remove_selected_images)
        self.remove_button.setToolTip("Drop the selected images and their annotations from the project.\n"
                                      "Takes effect when the project is saved.")

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.accept)

        button_layout.addWidget(self.search_button)
        button_layout.addWidget(self.locate_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.continue_button)

        self.layout().addLayout(button_layout)

    def refresh(self):
        """Sync the list, label and buttons with the images still missing."""
        self.list_widget.clear()
        for path in self.missing_images:
            self.list_widget.addItem(path)
        if self.missing_images:
            self.list_widget.setCurrentRow(0)

        missing = len(self.missing_images)
        removed = len(self.removed_images)
        found = self.total_missing - missing - removed
        text = f"{missing} of {self.total_missing} image(s) could not be found."
        if found:
            text += f" {found} found so far."
        if removed:
            text += f" {removed} to be removed."
        self.info_label.setText(text)
        self.update_buttons()

    def selected_rows(self):
        """Rows of the selected missing images, in list order."""
        return sorted(index.row() for index in self.list_widget.selectedIndexes())

    def update_buttons(self):
        """Enable buttons for what can be done now and label Continue with its cost."""
        missing = len(self.missing_images)
        selected = len(self.selected_rows())
        self.search_button.setEnabled(missing > 0)
        self.locate_button.setEnabled(selected == 1)
        self.remove_button.setEnabled(selected > 0)
        if missing:
            self.continue_button.setText(f"Continue without {missing}")
            self.continue_button.setToolTip("Open the project with the images that were found.\n"
                                            "The missing images and their annotations stay in the project file.")
        else:
            self.continue_button.setText("Continue")
            self.continue_button.setToolTip("Open the project.")

    def taken_paths(self):
        """Paths already in use: images that exist plus relinks already made."""
        taken = {path for path in self.image_paths if path not in self.updated_paths}
        taken.update(self.updated_paths.values())
        return taken

    def apply_matches(self, matches):
        """Record original -> new path relinks and drop them from the missing list."""
        for missing_path, new_path in matches.items():
            self.updated_paths[missing_path] = new_path
        self.missing_images = [path for path in self.missing_images if path not in matches]

    def resolved_image_paths(self):
        """The project's image paths with every relink applied."""
        return [self.updated_paths.get(path, path) for path in self.image_paths]

    def open_file_dialog(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(self, "Select Root Directory to Search", "")
        return directory if directory else None

    def find_matches(self, directory):
        """Walk a folder for files named like the missing images.

        When a name turns up more than once, the candidate sharing the most
        trailing folders with the old path wins. A tie, or a file another image
        already uses, is left missing rather than guessed.

        Returns:
            tuple: ({missing_path: new_path}, number of images left ambiguous)
        """
        targets = {}  # basename key -> [missing paths]
        for path in self.missing_images:
            targets.setdefault(_basename_key(path), []).append(path)

        candidates = {}  # basename key -> [found paths]
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                key = filename.lower()
                if key in targets:
                    candidates.setdefault(key, []).append(os.path.join(root, filename).replace("\\", "/"))

        taken = self.taken_paths()
        matches = {}
        ambiguous = 0
        for key, missing_paths in targets.items():
            found = [path for path in candidates.get(key, []) if path not in taken]
            if not found:
                continue
            found_parts = [(path, _path_parts(path)) for path in found]
            for missing_path in missing_paths:
                missing_parts = _path_parts(missing_path)
                scored = sorted(((_shared_tail(missing_parts, parts), path) for path, parts in found_parts),
                                reverse=True)
                if len(scored) > 1 and scored[0][0] == scored[1][0]:
                    ambiguous += 1
                    continue
                best = scored[0][1]
                matches[missing_path] = best
                taken.add(best)
                found_parts = [(path, parts) for path, parts in found_parts if path != best]
                if not found_parts:
                    break

        return matches, ambiguous

    def update_image_paths(self):
        """Update missing image paths using recursive search."""
        if not self.missing_images:
            return

        directory = self.open_file_dialog()
        if directory is None:  # User cancelled the folder picker; keep this dialog open
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            matches, ambiguous = self.find_matches(directory)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"An error occurred during the search: {str(e)}")
            return
        QApplication.restoreOverrideCursor()

        self.apply_matches(matches)
        self.refresh()

        if not self.missing_images:
            self.accept()
            return

        message = f"Found {len(matches)} image(s). {len(self.missing_images)} still missing."
        if ambiguous:
            message += (f"\n\n{ambiguous} image(s) matched more than one file equally well. "
                        f"Use Locate to pick the right one.")
        if matches:
            QMessageBox.information(self, "Images Updated", message)
        else:
            QMessageBox.warning(self, "No Images Found", message)

    def locate_selected_image(self):
        """Pick the file for the selected missing image by hand."""
        rows = self.selected_rows()
        if len(rows) != 1 or rows[0] >= len(self.missing_images):
            return
        row = rows[0]
        missing_path = self.missing_images[row]

        normalized = missing_path.replace("\\", "/")
        start_dir = self.last_locate_dir
        if not start_dir:
            parent_dir = os.path.dirname(normalized)
            start_dir = parent_dir if os.path.isdir(parent_dir) else ""

        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   f"Locate {normalized.rsplit('/', 1)[-1]}",
                                                   start_dir,
                                                   "All Files (*)")
        if not file_path:
            return
        file_path = file_path.replace("\\", "/")

        if file_path in self.taken_paths():
            QMessageBox.warning(self,
                                "Image Already In Project",
                                f"{file_path} is already used by another image in this project.")
            return

        self.last_locate_dir = os.path.dirname(file_path)
        self.apply_matches({missing_path: file_path})
        self.refresh()
        if self.missing_images:
            self.list_widget.setCurrentRow(min(row, len(self.missing_images) - 1))
        else:
            self.accept()

    def remove_selected_images(self):
        """Drop the selected missing images, and their annotations, from the project."""
        rows = [row for row in self.selected_rows() if row < len(self.missing_images)]
        if not rows:
            return
        paths = [self.missing_images[row] for row in rows]

        reply = QMessageBox.question(self,
                                     "Remove Images",
                                     f"Remove {len(paths)} image(s) and their annotations from the project?\n\n"
                                     f"The image files are not touched. The project file changes when you save it.",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        removed = set(paths)
        self.removed_images.extend(paths)
        self.missing_images = [path for path in self.missing_images if path not in removed]
        self.refresh()
        if self.missing_images:
            self.list_widget.setCurrentRow(min(rows[0], len(self.missing_images) - 1))
        else:
            self.accept()

    @staticmethod
    def update_paths(image_paths, parent=None):
        """Show the dialog and return what the project can open with.

        Closing the dialog counts as Continue: images still missing are reported
        back, never treated as a reason to drop the rest.

        Returns:
            tuple: (image paths with relinks applied, {original: new} relinks,
                    original paths still missing, original paths the user removed)
        """
        dialog = UpdateImagePaths(image_paths, parent)
        # The caller is usually inside a WaitCursor override; show a normal one here.
        QApplication.setOverrideCursor(Qt.ArrowCursor)
        try:
            dialog.exec_()
        finally:
            QApplication.restoreOverrideCursor()
        return (dialog.resolved_image_paths(), dict(dialog.updated_paths),
                list(dialog.missing_images), list(dialog.removed_images))
