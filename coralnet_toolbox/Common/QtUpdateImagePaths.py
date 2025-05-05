import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os 

from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QDoubleSpinBox, QComboBox, QSpinBox, QHBoxLayout,
                             QWidget, QStackedWidget, QGridLayout, QMessageBox,
                             QDialog, QListWidget, QPushButton, QFileDialog,
                             QGraphicsView)
from PyQt5.QtCore import Qt                           

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class UpdateImagePaths(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        
        self.setWindowIcon(get_icon("coral"))
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
        info_label = QLabel(f"The following {len(self.missing_images)} image(s) could not be found. "
                           "Please select a directory to search for these images.")
        info_label.setWordWrap(True)
        self.layout().addWidget(info_label)
    
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
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.cancel_button)
        
        self.layout().addLayout(button_layout)
    
    def open_file_dialog(self):
        """Open file selection dialog with first missing image name as filter."""
        if not self.missing_images:
            return None
            
        first_missing = self.missing_images[0]
        first_basename = os.path.basename(first_missing)
        
        # Create filter based on the file extension
        file_filter = f"Image files ({first_basename})"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select the corresponding file for '{first_basename}'",
            os.path.dirname(first_missing) if os.path.dirname(first_missing) else "",
            file_filter
        )
        
        return os.path.dirname(file_path) if file_path else None
    
    def update_image_paths(self):
        """Update missing image paths using file selection."""
        while self.missing_images:
            # Open file dialog and get the directory from selected file
            directory = self.open_file_dialog()
            if not directory:  # User cancelled
                return
            
            # Update paths for missing images
            updated_count = 0
            still_missing = []
            
            for missing_path in self.missing_images:
                basename = os.path.basename(missing_path)
                new_path = os.path.join(directory, basename)
                # Replace "\\" with "/" for Windows paths
                new_path = new_path.replace("\\", "/")
                
                if os.path.exists(new_path):
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
                QMessageBox.warning(self, "No Images Found", 
                                    "No images were found in the selected directory. "
                                    "Looking for files like: {first_basename}")
            elif still_missing:
                # Update the list widget with remaining missing images
                self.list_widget.clear()
                for path in still_missing:
                    self.list_widget.addItem(path)
                
                QMessageBox.information(self, "Images Updated", 
                                        "Updated {updated_count} image(s). "
                                        f"{len(still_missing)} image(s) still missing.")
            else:
                QMessageBox.information(self, "All Images Updated", 
                                        f"Successfully updated all {updated_count} missing image(s).")
                self.accept()
    
    @staticmethod
    def update_paths(image_paths, parent=None):
        """Static method to create and execute the dialog."""
        dialog = UpdateImagePaths(image_paths, parent)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            return dialog.image_paths, dialog.updated_paths
        else:
            return image_paths, {}  # Return original paths if cancelled