import warnings

import os
import gc
import ujson as json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QVBoxLayout, QLabel, QDialog,
                             QPushButton, QGroupBox, QHBoxLayout, QTableWidget, 
                             QTableWidgetItem)

from torch.cuda import empty_cache

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Base(QDialog):
    """
    Base class for deploying machine learning models.

    :param main_window: MainWindow object
    :param parent: Parent widget
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window
        self.image_window = main_window.image_window
        self.annotation_window = main_window.annotation_window
        self.sam_dialog = None

        self.setWindowIcon(get_icon("coralnet.png"))
        self.setWindowTitle("Deploy Model")
        self.resize(450, 650)

        # Initialize variables
        self.imgsz = 1024
        self.max_detect = 300
        self.uncertainty_thresh = 0.30
        self.iou_thresh = 0.20
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40
        
        self.BATCH_SIZE = 16

        self.task = None
        self.model_path = None
        self.loaded_model = None
        self.class_names = []
        self.class_mapping = {}
        self.auto_created_labels = set()  # Track which labels were auto-created
        self.label_to_class_name = {}  # Map row index to class name for checkbox tracking

        self.layout = QVBoxLayout(self)

        # Setup the info layout
        self.setup_info_layout()
        # Setup the labels layout
        self.setup_labels_layout()
        # Setup parameters layout
        self.setup_parameters_layout()
        # Setup SAM layout
        self.setup_sam_layout()
        # Setup thresholds layout
        self.setup_thresholds_layout()
        # Setup the button layout
        self.setup_buttons_layout()
        # Setup the status layout
        self.setup_status_layout()

    def setup_info_layout(self):
        """Set up the layout and widgets for the info layout."""
        group_box = QGroupBox("Information")
        layout = QVBoxLayout()

        # Create a QLabel with explanatory text and hyperlink
        info_label = QLabel("Deploy an Ultralytics model to use.")

        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_labels_layout(self):
        """Setup the labels layout with a table"""
        group_box = QGroupBox("Labels")
        layout = QVBoxLayout()

        # Create a table widget to display labels
        self.labels_table = QTableWidget()
        self.labels_table.setColumnCount(4)
        self.labels_table.setHorizontalHeaderLabels(["✓", "Status", "Short Label", "Long Label"])
        self.labels_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.labels_table.horizontalHeader().setStretchLastSection(True)
        self.labels_table.setColumnWidth(0, 50)
        self.labels_table.setColumnWidth(1, 60)
        self.labels_table.setColumnWidth(2, 120)
        layout.addWidget(self.labels_table)

        # Add status label
        self.labels_status_label = QLabel("No model file selected")
        layout.addWidget(self.labels_status_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_parameters_layout(self):
        raise NotImplementedError("Subclasses must implement this method")

    def setup_sam_layout(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def setup_thresholds_layout(self):
        raise NotImplementedError("Subclasses must implement this method")

    def setup_buttons_layout(self):
        """Set up the buttons layout in a 2x2 grid"""
        # Model controls group
        group_box = QGroupBox("Actions")
        layout = QVBoxLayout()  # Main vertical layout

        # Create two horizontal layouts for each row
        top_row = QHBoxLayout()
        bottom_row = QHBoxLayout()

        # Model control buttons
        self.browse_button = QPushButton("Browse Model")
        self.browse_button.clicked.connect(self.browse_file)

        self.mapping_button = QPushButton("Browse Class Mapping")
        self.mapping_button.clicked.connect(self.browse_class_mapping_file)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)

        self.deactivate_button = QPushButton("Deactivate Model")
        self.deactivate_button.clicked.connect(self.deactivate_model)

        # Add buttons to rows
        top_row.addWidget(self.browse_button)
        top_row.addWidget(self.mapping_button)
        bottom_row.addWidget(self.load_button)
        bottom_row.addWidget(self.deactivate_button)

        # Add rows to main layout
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_status_layout(self):
        """Setup the status layout"""
        # Create a group box for the status bar
        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        # Status bar for model status
        self.status_bar = QLabel("No model loaded")
        layout.addWidget(self.status_bar)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def is_sam_model_deployed(self):
        """
        Check if the SAM model is deployed and update the checkbox state accordingly.

        :return: Boolean indicating whether the SAM model is deployed
        """
        if not hasattr(self.main_window, 'sam_deploy_predictor_dialog'):
            return False

        self.sam_dialog = self.main_window.sam_deploy_predictor_dialog

        if not self.sam_dialog.loaded_model:
            self.use_sam_dropdown.setCurrentText("False")
            QMessageBox.critical(self, "Error", "Please deploy the SAM model first.")
            return False

        return True

    def browse_file(self):
        """Browse and select a model file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Model File", "",
            "Model Files (*.pt *.engine)",
            options=options
        )

        if file_path:
            # Clear the class mapping
            self.class_mapping = {}

            if ".bin" in file_path:  # TODO remove this
                # OpenVINO is a directory
                file_path = os.path.dirname(file_path)

            self.model_path = file_path
            self.labels_status_label.setText("Model file selected")

            # Try to load the class mapping file if it exists
            parent_dir = os.path.dirname(os.path.dirname(file_path))
            class_mapping_path = os.path.join(parent_dir, "class_mapping.json")
            if os.path.exists(class_mapping_path):
                self.load_class_mapping(class_mapping_path)

    def browse_class_mapping_file(self):
        """Browse and select a class mapping file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Class Mapping File", "",
            "JSON Files (*.json)",
            options=options
        )
        if file_path:
            self.load_class_mapping(file_path)

    def load_class_mapping(self, file_path):
        """
        Load the class mapping file

        :param file_path: Path to the class mapping file
        """
        try:
            with open(file_path, 'r') as f:
                self.class_mapping = json.load(f)  # maybe remove background? Already done in Semantic and works?
            self.labels_status_label.setText(self.labels_status_label.text() + " | Class mapping loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load class mapping file: {str(e)}")

    def _find_unmapped_classes(self):
        """
        Find classes in the model that are not present in the class mapping.
        
        :return: Tuple of (mapped_classes, unmapped_classes, unused_mapping_keys)
                 - mapped_classes: List of class names that exist in both model and mapping
                 - unmapped_classes: List of class names in model but not in mapping
                 - unused_mapping_keys: List of keys in mapping but not in model
        """
        if not self.class_names:
            return [], [], list(self.class_mapping.keys())
        
        model_names_set = set(self.class_names)
        mapping_keys_set = set(self.class_mapping.keys())
        
        mapped_classes = list(model_names_set & mapping_keys_set)
        unmapped_classes = list(model_names_set - mapping_keys_set)
        unused_mapping_keys = list(mapping_keys_set - model_names_set)
        
        return mapped_classes, unmapped_classes, unused_mapping_keys

    def load_model(self):
        """
        Load the model
        """
        raise NotImplementedError("Subclasses must implement this method")

    def check_and_display_class_names(self):
        """
        Check and display the class names with their mapping status in a table.
        Shows which labels are mapped from file, auto-created, or missing.
        """
        if not self.loaded_model:
            return

        missing_labels = []
        mapped_count = 0
        auto_created_count = 0
        missing_count = 0

        # Clear the table and set row count
        self.labels_table.setRowCount(len(self.class_names))
        self.label_to_class_name = {}  # Reset the mapping

        for row, class_name in enumerate(self.class_names):
            status_emoji = ""
            status_text = ""
            short_label = ""
            long_label = ""

            if class_name in self.auto_created_labels:
                # Auto-created label
                status_emoji = "⚠️"
                status_text = "Auto-created"
                short_label = class_name
                long_label = class_name
                auto_created_count += 1
            else:
                # Check if it exists in project labels
                label = self.label_window.get_label_by_short_code(class_name)  
                if label.id:
                    # Found in project labels (from mapping file or previous creation)
                    status_emoji = "✅"
                    status_text = "Mapped"
                    short_label = label.short_label_code
                    long_label = label.long_label_code
                    mapped_count += 1
                else:
                    # Not found anywhere
                    status_emoji = "❌"
                    status_text = "Missing"
                    short_label = class_name
                    long_label = ""
                    missing_labels.append(class_name)
                    missing_count += 1

            # Store mapping of row to class name for checkbox tracking
            self.label_to_class_name[row] = class_name

            # Add checkbox in first column (initially checked and disabled)
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(checkbox_item.flags() | Qt.ItemIsUserCheckable)
            flags = checkbox_item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled
            checkbox_item.setFlags(flags)
            checkbox_item.setCheckState(Qt.Checked)
            checkbox_item.setTextAlignment(Qt.AlignCenter)
            self.labels_table.setItem(row, 0, checkbox_item)

            # Add items to table with status in column 1
            status_item = QTableWidgetItem(status_emoji)
            status_item.setToolTip(status_text)
            status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.labels_table.setItem(row, 1, status_item)
            
            short_label_item = QTableWidgetItem(short_label)
            short_label_item.setToolTip(f"Short Label: {short_label}")
            short_label_item.setFlags(short_label_item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)
            short_label_item.setTextAlignment(Qt.AlignCenter)
            self.labels_table.setItem(row, 2, short_label_item)
            
            long_label_item = QTableWidgetItem(long_label)
            long_label_item.setToolTip(f"Long Label: {long_label}")
            long_label_item.setFlags(long_label_item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsSelectable)
            long_label_item.setTextAlignment(Qt.AlignCenter)
            self.labels_table.setItem(row, 3, long_label_item)

        # Show warning if there are missing labels
        if missing_labels:
            missing_labels_str = "\n".join(missing_labels)
            QMessageBox.warning(
                self,
                "Warning",
                f"The following {len(missing_labels)} class(es) are missing and cannot be predicted "
                f"until added manually:\n\n{missing_labels_str}"
            )

    def add_labels_to_label_window(self):
        """
        Add labels to the label window based on the class mapping.
        """
        if self.class_mapping:
            for label in self.class_mapping.values():
                self.label_window.add_label_if_not_exists(
                    label['short_label_code'], 
                    label['long_label_code'],
                    QColor(*label.get('color', [255, 255, 255])),
                    label_id=label.get('id')
                )

    def handle_missing_class_mapping(self, unmapped_classes=None):
        """
        Handle missing or incomplete class mappings.
        
        :param unmapped_classes: Optional list of class names missing from the mapping.
                                If None, all classes are treated as unmapped (no mapping file).
                                If provided, only these classes are unmapped (partial mapping).
        """
        if unmapped_classes is None:
            # No mapping file at all - offer to create generic labels for all classes
            reply = QMessageBox.question(
                self,
                'No Class Mapping Found',
                'Do you want to create generic labels automatically?',
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.create_generic_labels()
        else:
            # Partial mapping - some classes are missing
            unmapped_str = "\n".join(unmapped_classes)
            message = (
                f'The following {len(unmapped_classes)} class(es) from the model are not in the '
                f'class mapping file:\n\n{unmapped_str}\n\n'
                f'Do you want to create generic labels for these classes?'
            )
            
            reply = QMessageBox.question(
                self,
                'Incomplete Class Mapping',
                message,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No
            )
            
            # Yes = Create generic labels for unmapped classes
            if reply == QMessageBox.Yes:
                self.create_generic_labels(unmapped_classes)
            # No = Skip (keep only mapped classes)
            elif reply == QMessageBox.No:
                pass  # Do nothing, keep only mapped classes in mapping
            # Cancel = Abort the load
            else:
                raise RuntimeError("Model load cancelled by user due to incomplete class mapping")

    def create_generic_labels(self, class_names=None):
        """
        Create generic labels for the given class names.
        
        :param class_names: Optional list of class names to create labels for. 
                           If None, uses self.class_names
        """
        names_to_create = class_names if class_names is not None else self.class_names
        
        for class_name in names_to_create:
            # Create the label in the label window
            label = self.label_window.add_label_if_not_exists(
                class_name,
                class_name,
            )
            self.class_mapping[class_name] = label.to_dict()
            self.auto_created_labels.add(class_name)  # Track as auto-created

    def get_checked_class_names(self):
        """
        Get a list of class names that are currently checked in the table.
        
        :return: List of class names that have checked checkboxes
        """
        checked_classes = []
        for row in range(self.labels_table.rowCount()):
            checkbox_item = self.labels_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.Checked:
                if row in self.label_to_class_name:
                    checked_classes.append(self.label_to_class_name[row])
        return checked_classes

    def get_checked_labels(self):
        """
        Get a list of label dictionaries for all checked rows in the table.
        Each dictionary contains the label mapping information.
        
        :return: List of label dictionaries for checked class names
        """
        checked_labels = []
        for class_name in self.get_checked_class_names():
            if class_name in self.class_mapping:
                checked_labels.append(self.class_mapping[class_name])
        return checked_labels

    def predict(self, inputs):
        """
        Predict using deployed model
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def deactivate_model(self):
        """
        Deactivate the current model
        """
        self.loaded_model = None
        self.model_path = None
        self.class_mapping = None
        self.auto_created_labels = set()
        gc.collect()
        empty_cache()
        self.status_bar.setText("No model loaded")
        self.labels_status_label.setText("No model file selected")
        self.labels_table.setRowCount(0)
