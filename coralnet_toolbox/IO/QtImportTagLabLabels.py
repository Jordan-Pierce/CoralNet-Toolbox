import json
import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

from coralnet_toolbox.QtLabelWindow import Label

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImportTagLabLabels:
    def __init__(self, main_window):
        self.main_window = main_window
        self.label_window = main_window.label_window

    def import_taglab_labels(self):
        self.main_window.untoggle_all_tools()

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.label_window,
                                                   "Import TagLab Labels",
                                                   "",
                                                   "JSON Files (*.json);;All Files (*)",
                                                   options=options)
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                if 'Labels' in data:
                    data['labels'] = data.pop('Labels')
                elif 'labels' in data:
                    pass
                else:
                    QMessageBox.warning(self.label_window,
                                        "Invalid JSON Format",
                                        "The selected JSON file does not contain 'Labels' or 'labels' key.")
                    return
                                
                # Make cursor busy
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                # Create a progress bar
                total_labels = len(data['labels'])
                progress_bar = ProgressBar(self.label_window, "Importing TagLab Labels")
                progress_bar.show()
                progress_bar.start_progress(total_labels)

                for label_info in data['labels']:
                    try:
                        short_label_code = label_info['name'].strip()
                        long_label_code = label_info['name'].strip()
                        color = label_info['fill']
                        
                        # Create a QtColor object from the color string
                        color = QColor(color[0], color[1], color[2])

                        # Add label if it does not exist
                        self.label_window.add_label_if_not_exists(short_label_code, long_label_code, color)
                        
                    except Exception as e:
                        print(f"Warning: Could not import label {label_info['name']}: {str(e)}")
                        
                    # Update the progress bar
                    progress_bar.update_progress()

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "TagLab labels have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing TagLab labels: {str(e)}")
                
            finally:
                # Stop the progress bar
                progress_bar.stop_progress()
                progress_bar.close()
                QApplication.restoreOverrideCursor()
