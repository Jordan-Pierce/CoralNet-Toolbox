import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFileDialog, QApplication, QMessageBox, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton,
                             QComboBox, QFormLayout, QGroupBox, QScrollArea, QWidget)

from ultralytics import YOLO

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
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
        
        self.setWindowIcon(get_icon("coral.png"))
        self.setWindowTitle("Optimize Model")
        self.resize(400, 325)

        self.model_path = ""
        self.custom_params = []

        self.layout = QVBoxLayout(self)
        
        # Setup the information layout
        self.setup_info_layout()
        # Setup the model layout
        self.setup_model_layout()
        # Setup the parameters layout
        self.setup_parameters_layout()
        # Setup the buttons layout
        self.setup_buttons_layout()
        
    def setup_info_layout(self):
        """Setup information layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Details on different production formats can be found "
                            "<a href='https://docs.ultralytics.com/modes/export/#export-formats'>here</a>.")
        
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        
    def setup_model_layout(self):
        """Setup model layout."""
        group_box = QGroupBox("Model")
        layout = QFormLayout()

        # Model file selection layout
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_browse_button = QPushButton("Browse...")
        self.model_browse_button.clicked.connect(self.browse_file)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_browse_button)
        layout.addRow("Model Path:", model_layout)
        
        # Export Format Dropdown
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["torchscript", "onnx", "openvino", "engine"])
        self.export_format_combo.setEditable(True)
        layout.addRow("Export Format:", self.export_format_combo)

        # Set the layout to the group box and add to main layout
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
    
    def setup_parameters_layout(self):
        """Setup parameters layout."""
        # Create parameters group box
        group_box = QGroupBox("Parameters")
        group_layout = QVBoxLayout(group_box)

        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)

        # Custom parameters section (inside form layout)
        self.custom_params_layout = QVBoxLayout()
        form_layout.addLayout(self.custom_params_layout)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(form_widget)
        group_layout.addWidget(scroll_area)

        # Buttons for parameter management (outside scroll area)
        buttons_layout = QHBoxLayout()
        self.add_param_button = QPushButton("Add Parameter")
        self.add_param_button.clicked.connect(self.add_parameter_pair)
        remove_button = QPushButton("Remove Parameter")
        remove_button.clicked.connect(self.remove_last_parameter)
        buttons_layout.addWidget(self.add_param_button)
        buttons_layout.addWidget(remove_button)
        group_layout.addLayout(buttons_layout)

        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Setup buttons layout."""
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(cancel_button)
        
        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.optimize_model)
        self.layout.addWidget(accept_button)

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
        
    def remove_last_parameter(self):
        """
        Remove the last parameter pair from the layout.
        """
        if self.custom_params:
            # Get the last parameter pair
            param_name, param_value = self.custom_params.pop()
            # Get the layout containing these widgets
            layout_to_remove = self.custom_params_layout.takeAt(self.custom_params_layout.count() - 1)
            # Delete the widgets
            while layout_to_remove.count():
                widget = layout_to_remove.takeAt(0).widget()
                widget.deleteLater()
            # Delete the layout
            layout_to_remove.deleteLater()

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
            if ".pt" in file_path or ".pth" in file_path:
                self.model_path = file_path
                self.model_path_edit.setText(file_path)
            else:
                QMessageBox.critical(self, "Error", "Please select a .pt model file.")
                self.model_path_edit.setText("")

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