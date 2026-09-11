import warnings

import os 

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QHBoxLayout, QApplication,
                             QMessageBox, QDialog, QListWidget, QPushButton, QFileDialog)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.Icons import get_window_icon


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class UpdateImagePaths(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.main_window = parent
        
        self.setWindowIcon(get_window_icon("coral"))
        self.setWindowTitle("Update Image Paths")
        self.resize(400, 300)
        
        # Original Image Paths
        self.image_paths = image_paths
        self.existing_images = self.filter_existing_images()
        self.missing_images = self.find_missing_images()
        
        # Mapping between original and updated paths
        self.updated_paths = {}
        
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        
        # Setup the information layout
        self.setup_info_layout()
        # Setup the list layout
        self.setup_list_widget()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def filter_existing_images(self):
        """Returns a list of image paths that exist."""
        return [path for path in self.image_paths if os.path.exists(path)]
    
    def find_missing_images(self):
        """Returns a list of image paths that don't exist."""
        return [path for path in self.image_paths if not os.path.exists(path)]
    
    def setup_info_layout(self):
        """Create information label explaining missing images."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        info_label = QLabel("The following images could not be found. "
                            "Please select a root directory; we will recursively search all subfolders for matches.")
        info_label.setWordWrap(True)
        info_label.setToolTip("Some images in the project cannot be highly located.\nSelect a directory to search for them by filename.")
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout().addWidget(group_box)
    
    def setup_list_widget(self):
        """Create list widget showing missing image paths."""
        self.list_widget = QListWidget()
        for path in self.missing_images:
            self.list_widget.addItem(path)
        self.layout().addWidget(self.list_widget)
    
    def setup_buttons_layout(self):
        """Create Update and Cancel buttons."""
        button_layout = QHBoxLayout()

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.update_image_paths)
        self.update_button.setToolTip("Search for missing images in a selected directory.\nImages will be matched by filename.")

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setToolTip("Close this dialog without updating image paths.")

        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.cancel_button)

        self.layout().addLayout(button_layout)
    
    def open_file_dialog(self):
        """Open directory selection dialog."""
        if not self.missing_images:
            return None
            
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Root Directory to Search",
            ""
        )
        
        return directory if directory else None
    
    def update_image_paths(self):
        """Update missing image paths using recursive search."""
        if not self.missing_images:
            return

        # Start progress bar
        progress_bar = ProgressBar(self, title="Searching for images...")
        progress_bar.show()
        
        try:
            directory = self.open_file_dialog()
            if directory is None:  # User cancelled or closed dialog
                self.reject()
                return

            # Prepare for recursive search
            # We use a set for O(1) lookups of basenames
            target_basenames = {os.path.basename(p) for p in self.missing_images}
            
            # Track which ones we've found
            found_map = {} # basename -> new_full_path
            
            # We'll use the progress bar to show how many files we've checked
            # But since we don't know how many files are in the tree, 
            # we'll use the "busy" mode or just update the title.
            progress_bar.set_busy_mode("Searching...")

            # Perform the recursive search
            for root, dirs, files in os.walk(directory):
                # Check files in the current directory
                for filename in files:
                    if filename in target_basenames:
                        new_path = os.path.join(root, filename).replace("\\", "/")
                        found_map[filename] = new_path
                
                # Check if we've found everything
                if len(found_map) == len(target_basenames):
                    break

            # Now update the actual paths
            updated_count = 0
            still_missing = []
            
            for missing_path in self.missing_images:
                basename = os.path.basename(missing_path)
                if basename in found_map:
                    new_path = found_map[basename]
                    # Update the path in the main image_paths list
                    index = self.image_paths.index(missing_path)
                    self.image_paths[index] = new_path
                    # Update the mapping between original and updated paths
                    self.updated_paths[missing_path] = new_path
                    updated_count += 1
                else:
                    still_missing.append(missing_path)
            
            # Update the missing images list
            self.missing_images = still_missing

            if updated_count == 0:
                QMessageBox.warning(self, 
                                    "No Images Found",
                                    "No matching images were found in the selected directory or its subfolders.")
            elif still_missing:
                # Update the list widget with remaining missing images
                self.list_widget.clear()
                for path in still_missing:
                    self.list_widget.addItem(path)
                
                QMessageBox.information(self, 
                                        "Images Updated", 
                                        f"Updated {updated_count} image(s). "
                                        f"{len(still_missing)} image(s) still missing.")
            else:
                QMessageBox.information(self, 
                                        "All Images Updated", 
                                        f"Successfully updated all {updated_count} missing image(s).")
                self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during the search: {int(e)}")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
            # Stop the progress bar
            progress_bar.stop_progress()
            progress_bar.close()

    
    @staticmethod
    def update_paths(image_paths, parent=None):
        """Static method to create and execute the dialog."""
        dialog = UpdateImagePaths(image_paths, parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.image_paths, dialog.updated_paths
        else:
            # User cancelled the update dialog — signal caller to abort import by
            # returning None for image_paths.
            return None, {}
