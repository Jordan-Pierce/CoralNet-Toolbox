import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QTextEdit, QPushButton, QComboBox, QFormLayout)

from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class OptimizeModelDialog(QDialog):
    """
    A dialog for optimizing and exporting a YOLO model with custom parameters.
    """

    def __init__(self, main_window, parent=None):
        """
        Initialize the dialog with the main window and optional parent widget.

        param main_window: The main window of the application.
        param parent: The parent widget of the dialog.
        """
        super().__init__(parent)
        self.main_window = main_window

        self.custom_params = []

        self.setWindowTitle("Optimize Model")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different production formats can be found "
                            "<a href='https://docs.ultralytics.com/modes/export/#export-formats'>here</a>.")
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        self.layout.addWidget(info_label)

        # Button to browse for model file
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        self.layout.addWidget(browse_button)

        # Text area to display selected model file path
        self.model_text_area = QTextEdit("No model file selected")
        self.model_text_area.setReadOnly(True)
        self.layout.addWidget(self.model_text_area)

        # Export Format Dropdown
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["torchscript", "onnx", "openvino", "engine"])
        self.export_format_combo.setEditable(True)
        self.layout.addWidget(QLabel("Select or Enter Export Format:"))
        self.layout.addWidget(self.export_format_combo)

        # Parameters Form
        self.form_layout = QFormLayout()

        # Add custom parameters section
        self.custom_params_layout = QVBoxLayout()
        self.form_layout.addRow("Parameters:", self.custom_params_layout)

        # Add button for new parameter pairs
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        self.form_layout.addRow("", self.add_param_button)

        self.layout.addLayout(self.form_layout)

        # Accept button to start optimization
        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.optimize_model)
        self.layout.addWidget(accept_button)

        self.setLayout(self.layout)

    def add_parameter_pair(self):
        """
        Add a new pair of parameter name and value input fields.
        """
        param_layout = QHBoxLayout()
        param_name = QLineEdit()
        param_value = QLineEdit()
        param_layout.addWidget(param_name)
        param_layout.addWidget(param_value)

        self.custom_params.append((param_name, param_value))
        self.custom_params_layout.addLayout(param_layout)

    def browse_file(self):
        """
        Open a file dialog to select a model file and display its path.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                   "Open Model File", 
                                                   "", 
                                                   "Model Files (*.pt)", 
                                                   options=options)
        if file_path:
            self.model_path = file_path
            self.model_text_area.setText("Model file selected")

    def accept(self):
        """
        Override the accept method to optimize the model before closing the dialog.
        """
        self.optimize_model()
        super().accept()

    def get_optimization_parameters(self):
        """
        Extract and return the optimization parameters from the dialog widgets.
        """
        params = {'format': self.export_format_combo.currentText()}

        for param_name, param_value in self.custom_params:
            name = param_name.text().strip()
            value = param_value.text().strip().lower()
            if name:
                if value == 'true':
                    params[name] = True
                elif value == 'false':
                    params[name] = False
                else:
                    try:
                        params[name] = int(value)
                    except ValueError:
                        try:
                            params[name] = float(value)
                        except ValueError:
                            params[name] = value

        return params

    def optimize_model(self):
        """
        Optimize and export the model using the specified parameters.
        """
        params = self.get_optimization_parameters()

        # Set the cursor to waiting (busy) cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # Initialize the model and export with given params
            YOLO(self.model_path).export(**params)

            message = "Model export successful."
            QMessageBox.information(self, "Model Export Status", message)

        except Exception as e:
            # Display an error message box to the user
            error_message = f"An error occurred when converting model: {e}"
            QMessageBox.critical(self, "Error", error_message)
            print(error_message)

        # Restore the cursor to the default cursor
        QApplication.restoreOverrideCursor()
