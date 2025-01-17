import json
import warnings

from PyQt5.QtWidgets import QFileDialog, QMessageBox

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
                
                # Create a progress bar
                total_labels = len(data['labels'])
                progress_bar = ProgressBar("Importing TagLab Labels", self.label_window)
                progress_bar.show()
                progress_bar.start_progress(total_labels)

                for label_id, label_info in data['labels'].items():
                    short_label_code = label_info['name'].strip()
                    long_label_code = label_info['name'].strip()
                    color = label_info['fill']

                    label = Label(short_label_code, long_label_code, color, label_id)
                    if not self.label_window.label_exists(label.short_label_code, label.long_label_code):
                        self.label_window.add_label(label.short_label_code,
                                                    label.long_label_code,
                                                    label.color,
                                                    label.id)
                        
                    # Update the progress bar
                    progress_bar.update_progress()
                    
                # Close the progress bar
                progress_bar.close()
                progress_bar.stop_progress()

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "TagLab labels have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing TagLab labels: {str(e)}")
