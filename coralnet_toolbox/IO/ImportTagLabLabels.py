import json
import warnings

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from coralnet_toolbox.QtLabelWindow import Label

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

                if 'labels' not in data:
                    QMessageBox.warning(self.label_window,
                                        "Invalid JSON Format",
                                        "The selected JSON file does not contain 'labels' key.")
                    return

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

                QMessageBox.information(self.label_window,
                                        "Labels Imported",
                                        "TagLab labels have been successfully imported.")

            except Exception as e:
                QMessageBox.warning(self.label_window,
                                    "Error Importing Labels",
                                    f"An error occurred while importing TagLab labels: {str(e)}")
