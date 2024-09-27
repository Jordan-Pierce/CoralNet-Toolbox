import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy, QMessageBox,
                             QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox, QSlider, QDoubleSpinBox)

from torch.cuda import is_available, device_count

from toolbox.QtAnnotation import PatchSamplingDialog
from toolbox.QtAnnotation import AnnotationWindow
from toolbox.QtConfidence import ConfidenceWindow
from toolbox.QtEventFilter import GlobalEventFilter
from toolbox.QtImage import ImageWindow
from toolbox.QtLabel import LabelWindow
from toolbox.QtMachineLearning import BatchInferenceDialog
from toolbox.QtMachineLearning import CreateDatasetDialog
from toolbox.QtMachineLearning import DeployModelDialog
from toolbox.QtMachineLearning import EvaluateModelDialog
from toolbox.QtMachineLearning import MergeDatasetsDialog
from toolbox.QtMachineLearning import OptimizeModelDialog
from toolbox.QtMachineLearning import TrainModelDialog
from toolbox.utilities import get_icon_path

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state
    uncertaintyChanged = pyqtSignal(float)  # Signal to emit the current uncertainty threshold

    def __init__(self):
        super().__init__()

        # Define the icon path
        self.setWindowTitle("CoralNet-Toolbox")
        # Set the window icon
        main_window_icon_path = get_icon_path("coral.png")
        self.setWindowIcon(QIcon(main_window_icon_path))

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)

        self.annotation_window = AnnotationWindow(self)
        self.label_window = LabelWindow(self)
        self.image_window = ImageWindow(self)
        self.confidence_window = ConfidenceWindow(self)

        # Set the default uncertainty threshold for Deploy Model and Batch Inference
        self.uncertainty_thresh = 0.1

        self.create_dataset_dialog = CreateDatasetDialog(self)
        self.merge_datasets_dialog = MergeDatasetsDialog(self)
        self.train_model_dialog = TrainModelDialog(self)
        self.evaluate_model_dialog = EvaluateModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.deploy_model_dialog = DeployModelDialog(self)
        self.batch_inference_dialog = BatchInferenceDialog(self)

        self.patch_sampling_dialog = PatchSamplingDialog(self)

        # Connect signals to update status bar
        self.annotation_window.imageLoaded.connect(self.update_image_dimensions)
        self.annotation_window.mouseMoved.connect(self.update_mouse_position)

        # Connect the toolChanged signal (to the AnnotationWindow)
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        # Connect the toolChanged signal (to the Toolbar)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)
        # Connect the selectedLabel signal to the LabelWindow's set_selected_label method
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)
        # Connect the annotationSelected signal from AnnotationWindow to update the transparency slider
        self.annotation_window.transparencyChanged.connect(self.update_annotation_transparency)
        # Connect the labelSelected signal from LabelWindow to update the transparency slider
        self.label_window.transparencyChanged.connect(self.update_label_transparency)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Menu bar
        self.menu_bar = self.menuBar()

        # Import menu
        self.import_menu = self.menu_bar.addMenu("Import")

        self.import_images_action = QAction("Import Images", self)
        self.import_images_action.triggered.connect(self.image_window.import_images)
        self.import_menu.addAction(self.import_images_action)
        self.import_menu.addSeparator()

        self.import_labels_action = QAction("Import Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.label_window.import_labels)
        self.import_menu.addAction(self.import_labels_action)
        self.import_menu.addSeparator()

        self.import_annotations_action = QAction("Import Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.annotation_window.import_annotations)
        self.import_menu.addAction(self.import_annotations_action)

        self.import_coralnet_annotations_action = QAction("Import CoralNet Annotations (CSV)", self)
        self.import_coralnet_annotations_action.triggered.connect(self.annotation_window.import_coralnet_annotations)
        self.import_menu.addAction(self.import_coralnet_annotations_action)

        self.import_viscore_annotations_action = QAction("Import Viscore Annotations (CSV)", self)
        self.import_viscore_annotations_action.triggered.connect(self.annotation_window.import_viscore_annotations)
        self.import_menu.addAction(self.import_viscore_annotations_action)

        # Export menu
        self.export_menu = self.menu_bar.addMenu("Export")

        self.export_labels_action = QAction("Export Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.label_window.export_labels)
        self.export_menu.addAction(self.export_labels_action)
        self.export_menu.addSeparator()

        self.export_annotations_action = QAction("Export Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.annotation_window.export_annotations)
        self.export_menu.addAction(self.export_annotations_action)

        self.export_coralnet_annotations_action = QAction("Export CoralNet Annotations (CSV)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.annotation_window.export_coralnet_annotations)
        self.export_menu.addAction(self.export_coralnet_annotations_action)

        self.export_viscore_annotations_action = QAction("Export Viscore Annotations (JSON)", self)
        self.export_viscore_annotations_action.triggered.connect(self.annotation_window.export_viscore_annotations)
        self.export_menu.addAction(self.export_viscore_annotations_action)

        # Patch Annotations Sampling menu
        self.patch_sampling_action = QAction("Sample", self)
        self.patch_sampling_action.triggered.connect(self.open_patch_sampling_dialog)
        self.menu_bar.addAction(self.patch_sampling_action)

        # CoralNet menu
        self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        self.coralnet_authenticate_action = QAction("Authenticate", self)
        self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        self.coralnet_upload_action = QAction("Upload", self)
        self.coralnet_menu.addAction(self.coralnet_upload_action)

        self.coralnet_download_action = QAction("Download", self)
        self.coralnet_menu.addAction(self.coralnet_download_action)

        self.coralnet_model_api_action = QAction("Model API", self)
        self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

        self.ml_create_dataset_action = QAction("Create Dataset", self)
        self.ml_create_dataset_action.triggered.connect(self.open_create_dataset_dialog)
        self.ml_menu.addAction(self.ml_create_dataset_action)

        self.ml_merge_datasets_action = QAction("Merge Datasets", self)
        self.ml_merge_datasets_action.triggered.connect(self.open_merge_datasets_dialog)
        self.ml_menu.addAction(self.ml_merge_datasets_action)

        self.ml_train_model_action = QAction("Train Model", self)
        self.ml_train_model_action.triggered.connect(self.open_train_model_dialog)
        self.ml_menu.addAction(self.ml_train_model_action)

        self.ml_evaluate_model_action = QAction("Evaluate Model", self)
        self.ml_evaluate_model_action.triggered.connect(self.open_evaluate_model_dialog)
        self.ml_menu.addAction(self.ml_evaluate_model_action)

        self.ml_optimize_model_action = QAction("Optimize Model", self)
        self.ml_optimize_model_action.triggered.connect(self.open_optimize_model_dialog)
        self.ml_menu.addAction(self.ml_optimize_model_action)

        self.ml_deploy_model_action = QAction("Deploy Model", self)
        self.ml_deploy_model_action.triggered.connect(self.open_deploy_model_dialog)
        self.ml_menu.addAction(self.ml_deploy_model_action)

        self.ml_batch_inference_action = QAction("Batch Inference", self)
        self.ml_batch_inference_action.triggered.connect(self.open_batch_inference_dialog)
        self.ml_menu.addAction(self.ml_batch_inference_action)

        # Create and add the toolbar
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(40)
        self.toolbar.setMovable(False)  # Lock the toolbar in place
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        # Add a spacer before the first tool with a fixed height
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedHeight(10)  # Set a fixed height for the spacer
        self.toolbar.addWidget(spacer)

        # Define icon paths
        select_icon_path = get_icon_path("select.png")
        patch_icon_path = get_icon_path("patch.png")
        polygon_icon_path = get_icon_path("polygon.png")
        turtle_icon_path = get_icon_path("turtle.png")
        rabbit_icon_path = get_icon_path("rabbit.png")
        rocket_icon_path = get_icon_path("rocket.png")

        # Add tools here with icons
        self.select_tool_action = QAction(QIcon(select_icon_path), "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)

        self.patch_tool_action = QAction(QIcon(patch_icon_path), "Patch", self)
        self.patch_tool_action.setCheckable(True)
        self.patch_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.patch_tool_action)

        self.polygon_tool_action = QAction(QIcon(polygon_icon_path), "Polygon", self)
        self.polygon_tool_action.setCheckable(True)
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)

        # Add a spacer to push the device label to the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        # Add the device label widget as an action in the toolbar
        self.device, num_device = self.get_available_device()
        if num_device == 1:
            device_icon = QIcon(rabbit_icon_path)
            device_tooltip = f"cuda:{self.device}"
        elif num_device > 1:
            device_icon = QIcon(rocket_icon_path)
            device_tooltip = f"cuda:{self.device}"
        else:
            device_icon = QIcon(turtle_icon_path)
            device_tooltip = "cpu"

        # Create the device action with the appropriate icon
        device_action = QAction(device_icon, "", self)  # Empty string for the text
        self.device_tool_action = device_action
        self.device_tool_action.setCheckable(False)
        # Set the tooltip to show the value of self.device
        self.device_tool_action.setToolTip(device_tooltip)
        self.toolbar.addAction(self.device_tool_action)

        # Create status bar layout
        self.status_bar_layout = QHBoxLayout()

        # Labels for image dimensions and mouse position
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")

        # Set fixed width for labels to prevent them from resizing
        self.image_dimensions_label.setFixedWidth(150)
        self.mouse_position_label.setFixedWidth(150)

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_annotation_transparency)
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)

        # Spin box for Uncertainty threshold control
        self.uncertainty_thresh_spinbox = QDoubleSpinBox()
        self.uncertainty_thresh_spinbox.setRange(0.0, 1.0)  # Range is 0.0 to 1.0
        self.uncertainty_thresh_spinbox.setSingleStep(0.01)  # Step size for the spinbox
        self.uncertainty_thresh_spinbox.setValue(self.uncertainty_thresh)
        self.uncertainty_thresh_spinbox.valueChanged.connect(self.update_uncertainty_thresh)

        # Spin box for annotation size control
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(1000)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)

        # Add widgets to status bar layout
        self.status_bar_layout.addWidget(self.image_dimensions_label)
        self.status_bar_layout.addWidget(self.mouse_position_label)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Transparency:"))
        self.status_bar_layout.addWidget(self.transparency_slider)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Uncertainty Threshold:"))
        self.status_bar_layout.addWidget(self.uncertainty_thresh_spinbox)
        self.status_bar_layout.addWidget(QLabel("Annotation Size:"))
        self.status_bar_layout.addWidget(self.annotation_size_spinbox)

        # --------------------------------------------------
        # Create the main layout
        # --------------------------------------------------
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create left and right layouts
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Add status bar layout to left layout above the AnnotationWindow
        self.left_layout.addLayout(self.status_bar_layout)
        self.left_layout.addWidget(self.annotation_window, 85)
        self.left_layout.addWidget(self.label_window, 15)

        # Add widgets to right layout
        self.right_layout.addWidget(self.image_window, 54)
        self.right_layout.addWidget(self.confidence_window, 46)

        # Add left and right layouts to main layout
        self.main_layout.addLayout(self.left_layout, 85)
        self.main_layout.addLayout(self.right_layout, 15)

        # Set up global event filter
        self.global_event_filter = GlobalEventFilter(self)
        QApplication.instance().installEventFilter(self.global_event_filter)

    def showEvent(self, event):
        super().showEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                # Allow minimizing
                pass
            elif self.windowState() & Qt.WindowMaximized:
                # Window is maximized, do nothing
                pass
            else:
                # Restore to normal state
                pass  # Do nothing, let the OS handle the restore

    def get_available_device(self):
        device = ''
        if is_available():
            for i in range(device_count()):
                device += f'{i}'
        else:
            device = 'cpu'
        return device, device_count()

    def toggle_tool(self, state):
        action = self.sender()
        if action == self.select_tool_action:
            if state:
                self.patch_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.toolChanged.emit("select")
            else:
                self.toolChanged.emit(None)
        elif action == self.patch_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.toolChanged.emit("patch")
            else:
                self.toolChanged.emit(None)
        elif action == self.polygon_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.toolChanged.emit("polygon")
            else:
                self.toolChanged.emit(None)

    def untoggle_all_tools(self):
        self.select_tool_action.setChecked(False)
        self.patch_tool_action.setChecked(False)
        self.polygon_tool_action.setChecked(False)
        self.toolChanged.emit(None)

    def handle_tool_changed(self, tool):
        if tool == "select":
            self.select_tool_action.setChecked(True)
            self.patch_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
        elif tool == "patch":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(True)
            self.polygon_tool_action.setChecked(False)
        elif tool == "polygon":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(True)
        else:
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)

    def update_image_dimensions(self, width, height):
        self.image_dimensions_label.setText(f"Image: {width} x {height}")

    def update_mouse_position(self, x, y):
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")

    def update_annotation_transparency(self, value):
        self.annotation_window.update_annotation_transparency(value)
        self.update_transparency_slider(value)  # Update the slider value

    def update_label_transparency(self, value):
        self.label_window.update_label_transparency(value)
        self.update_transparency_slider(value)  # Update the slider value

    def update_transparency_slider(self, transparency):
        self.transparency_slider.setValue(transparency)

    def get_uncertainty_thresh(self):
        return self.uncertainty_thresh

    def update_uncertainty_thresh(self, value):
        if self.uncertainty_thresh != value:
            self.uncertainty_thresh = value
            self.uncertainty_thresh_spinbox.setValue(value)
            self.uncertaintyChanged.emit(value)

    def open_import_images_dialog(self):
        self.image_window.import_images()

    def open_patch_sampling_dialog(self):

        if not self.image_window.image_paths:
            # Check if there are any images in the project
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Proceed to open the dialog if images are loaded
            self.patch_sampling_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

        self.patch_sampling_dialog = None
        self.patch_sampling_dialog = PatchSamplingDialog(self)

    def open_create_dataset_dialog(self):
        # Check if there are loaded images
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Create Dataset",
                                "No images are present in the project.")
            return

        # Check if there are annotations
        if not len(self.annotation_window.annotations_dict):
            QMessageBox.warning(self,
                                "Create Dataset",
                                "No annotations are present in the project.")
            return

        try:
            self.create_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_merge_datasets_dialog(self):
        try:
            self.merge_datasets_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_train_model_dialog(self):
        try:
            self.train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_evaluate_model_dialog(self):
        try:
            self.evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        try:
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_deploy_model_dialog(self):
        try:
            self.deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        # Check if there are annotations
        if not len(self.annotation_window.annotations_dict):
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No annotations are present in the project.")
            return

        if not self.deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")