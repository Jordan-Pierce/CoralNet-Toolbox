import warnings

from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtWidgets import (QDoubleSpinBox, QListWidget)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy, QMessageBox,
                             QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox, QSlider, QDialog, QPushButton)

from toolbox.QtAnnotationWindow import AnnotationWindow
from toolbox.QtConfidenceWindow import ConfidenceWindow
from toolbox.QtEventFilter import GlobalEventFilter
from toolbox.QtImageWindow import ImageWindow
from toolbox.QtLabelWindow import LabelWindow
from toolbox.QtPatchSampling import PatchSamplingDialog

from toolbox.MachineLearning.QtBatchInference import BatchInferenceDialog
from toolbox.MachineLearning.QtImportDataset import ImportDatasetDialog
from toolbox.MachineLearning.QtExportDataset import ExportDatasetDialog
from toolbox.MachineLearning.QtDeployModel import DeployModelDialog
from toolbox.MachineLearning.QtEvaluateModel import EvaluateModelDialog
from toolbox.MachineLearning.QtMergeDatasets import MergeDatasetsDialog
from toolbox.MachineLearning.QtOptimizeModel import OptimizeModelDialog
from toolbox.MachineLearning.QtTrainModel import TrainModelDialog

from toolbox.QtSAM import SAMDeployModelDialog

from toolbox.QtIO import IODialog

from toolbox.utilities import get_available_device
from toolbox.utilities import get_icon_path

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state
    uncertaintyChanged = pyqtSignal(float)  # Signal to emit the current uncertainty threshold
    iouChanged = pyqtSignal(float)  # Signal to emit the current IoU threshold

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

        self.io_dialog = IODialog(self)

        # Set the default uncertainty threshold for Deploy Model and Batch Inference
        self.iou_thresh = 0.25
        self.uncertainty_thresh = 0.50

        self.import_dataset_dialog = ImportDatasetDialog(self)
        self.export_dataset_dialog = ExportDatasetDialog(self)
        self.merge_datasets_dialog = MergeDatasetsDialog(self)
        self.train_model_dialog = TrainModelDialog(self)
        self.evaluate_model_dialog = EvaluateModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.deploy_model_dialog = DeployModelDialog(self)
        self.batch_inference_dialog = BatchInferenceDialog(self)
        self.sam_deploy_model_dialog = SAMDeployModelDialog(self)
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)

        # Connect signals to update status bar
        self.annotation_window.imageLoaded.connect(self.update_image_dimensions)
        self.annotation_window.mouseMoved.connect(self.update_mouse_position)
        self.annotation_window.viewChanged.connect(self.update_view_dimensions)

        # Connect the hover_point signal from AnnotationWindow to the methods in SAMTool
        self.annotation_window.hover_point.connect(self.annotation_window.tools["sam"].start_hover_timer)
        self.annotation_window.hover_point.connect(self.annotation_window.tools["sam"].stop_hover_timer)

        # Connect the toolChanged signal (to the AnnotationWindow)
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        # Connect the toolChanged signal (to the Toolbar)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)
        # Connect the selectedLabel signal to the LabelWindow's set_selected_label method
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)
        # Connect the labelSelected signal from LabelWindow to update the transparency slider
        self.label_window.transparencyChanged.connect(self.update_label_transparency)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageChanged signal from ImageWindow to cancel SAM working area
        self.image_window.imageChanged.connect(self.handle_image_changed)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # Menu bar
        self.menu_bar = self.menuBar()

        # Import menu
        self.import_menu = self.menu_bar.addMenu("Import")

        # Raster submenu
        self.import_rasters_menu = self.import_menu.addMenu("Rasters")

        self.import_images_action = QAction("Images", self)
        self.import_images_action.triggered.connect(self.io_dialog.import_images)
        self.import_rasters_menu.addAction(self.import_images_action)

        self.import_ortho_action = QAction("Orthomosaic", self)
        self.import_ortho_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.import_rasters_menu.addAction(self.import_ortho_action)

        self.import_frames_action = QAction("Video Frames", self)
        self.import_frames_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.import_rasters_menu.addAction(self.import_frames_action)

        # Labels submenu
        self.import_labels_menu = self.import_menu.addMenu("Labels")

        self.import_labels_action = QAction("Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.io_dialog.import_labels)
        self.import_labels_menu.addAction(self.import_labels_action)

        # Annotations submenu
        self.import_annotations_menu = self.import_menu.addMenu("Annotations")

        self.import_annotations_action = QAction("Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.io_dialog.import_annotations)
        self.import_annotations_menu.addAction(self.import_annotations_action)

        self.import_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.import_coralnet_annotations_action.triggered.connect(self.io_dialog.import_coralnet_annotations)
        self.import_annotations_menu.addAction(self.import_coralnet_annotations_action)

        self.import_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.import_viscore_annotations_action.triggered.connect(self.io_dialog.import_viscore_annotations)
        self.import_annotations_menu.addAction(self.import_viscore_annotations_action)

        self.import_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.import_taglab_annotations_action.triggered.connect(self.io_dialog.import_taglab_annotations)
        self.import_annotations_menu.addAction(self.import_taglab_annotations_action)

        # Dataset submenu
        self.import_dataset_menu = self.import_menu.addMenu("Dataset")

        # Import YOLO Dataset menu
        self.import_dataset_action = QAction("YOLO (TXT)", self)
        self.import_dataset_action.triggered.connect(self.open_import_dataset_dialog)
        self.import_dataset_menu.addAction(self.import_dataset_action)

        # Export menu
        self.export_menu = self.menu_bar.addMenu("Export")

        # Labels submenu
        self.export_labels_menu = self.export_menu.addMenu("Labels")

        self.export_labels_action = QAction("Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.io_dialog.export_labels)
        self.export_labels_menu.addAction(self.export_labels_action)

        # Annotations submenu
        self.export_annotations_menu = self.export_menu.addMenu("Annotations")

        self.export_annotations_action = QAction("Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.io_dialog.export_annotations)
        self.export_annotations_menu.addAction(self.export_annotations_action)

        self.export_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.io_dialog.export_coralnet_annotations)
        self.export_annotations_menu.addAction(self.export_coralnet_annotations_action)

        self.export_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.export_viscore_annotations_action.triggered.connect(self.io_dialog.export_viscore_annotations)
        self.export_annotations_menu.addAction(self.export_viscore_annotations_action)

        self.export_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.export_taglab_annotations_action.triggered.connect(self.io_dialog.export_taglab_annotations)
        self.export_annotations_menu.addAction(self.export_taglab_annotations_action)

        # Dataset submenu
        self.export_dataset_menu = self.export_menu.addMenu("Dataset")

        # Export YOLO Dataset menu
        self.export_dataset_action = QAction("YOLO (TXT)", self)
        self.export_dataset_action.triggered.connect(self.open_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_dataset_action)

        # Sampling Annotations menu
        self.annotation_sampling_action = QAction("Sample", self)
        self.annotation_sampling_action.triggered.connect(self.open_patch_annotation_sampling_dialog)
        self.menu_bar.addAction(self.annotation_sampling_action)

        # CoralNet menu
        self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        self.coralnet_authenticate_action = QAction("Authenticate", self)
        self.coralnet_authenticate_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        self.coralnet_upload_action = QAction("Upload", self)
        self.coralnet_upload_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.coralnet_menu.addAction(self.coralnet_upload_action)

        self.coralnet_download_action = QAction("Download", self)
        self.coralnet_download_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.coralnet_menu.addAction(self.coralnet_download_action)

        self.coralnet_model_api_action = QAction("Model API", self)
        self.coralnet_model_api_action.triggered.connect(
            lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

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

        # SAM menu
        self.sam_menu = self.menu_bar.addMenu("SAM")

        self.sam_deploy_model_action = QAction("Deploy Model", self)
        self.sam_deploy_model_action.triggered.connect(self.open_sam_deploy_model_dialog)
        self.sam_menu.addAction(self.sam_deploy_model_action)

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
        self.select_icon_path = get_icon_path("select.png")
        self.patch_icon_path = get_icon_path("patch.png")
        self.rectangle_icon_path = get_icon_path("rectangle.png")
        self.polygon_icon_path = get_icon_path("polygon.png")
        self.sam_icon_path = get_icon_path("sam.png")
        self.turtle_icon_path = get_icon_path("turtle.png")
        self.rabbit_icon_path = get_icon_path("rabbit.png")
        self.rocket_icon_path = get_icon_path("rocket.png")
        self.apple_icon_path = get_icon_path("apple.png")

        # Add tools here with icons
        self.select_tool_action = QAction(QIcon(self.select_icon_path), "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)

        self.patch_tool_action = QAction(QIcon(self.patch_icon_path), "Patch", self)
        self.patch_tool_action.setCheckable(True)
        self.patch_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.patch_tool_action)

        self.rectangle_tool_action = QAction(QIcon(self.rectangle_icon_path), "Rectangle", self)
        self.rectangle_tool_action.setCheckable(True)
        self.rectangle_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.rectangle_tool_action)

        self.polygon_tool_action = QAction(QIcon(self.polygon_icon_path), "Polygon", self)
        self.polygon_tool_action.setCheckable(True)
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)

        self.sam_tool_action = QAction(QIcon(self.sam_icon_path), "SAM", self)
        self.sam_tool_action.setCheckable(True)
        self.sam_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.sam_tool_action)

        # Add a spacer to push the device label to the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        # Add the device label widget as an action in the toolbar
        self.devices = get_available_device()
        self.current_device_index = 0
        self.device = self.devices[self.current_device_index]

        if self.device.startswith('cuda'):
            if len(self.devices) == 1:
                device_icon = QIcon(self.rabbit_icon_path)
            else:
                device_icon = QIcon(self.rocket_icon_path)
            device_tooltip = self.device
        elif self.device == 'mps':
            device_icon = QIcon(self.apple_icon_path)
            device_tooltip = 'mps'
        else:
            device_icon = QIcon(self.turtle_icon_path)
            device_tooltip = 'cpu'

        # Create the device action with the appropriate icon
        device_action = ClickableAction(device_icon, "", self)  # Empty string for the text
        self.device_tool_action = device_action
        self.device_tool_action.setCheckable(False)
        # Set the tooltip to show the value of self.device
        self.device_tool_action.setToolTip(device_tooltip)
        self.device_tool_action.triggered.connect(self.toggle_device)
        self.toolbar.addAction(self.device_tool_action)

        # Create status bar layout
        self.status_bar_layout = QHBoxLayout()

        # Labels for image dimensions and mouse position
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.view_dimensions_label = QLabel("View: 0 x 0")  # Add QLabel for view dimensions

        # Set fixed width for labels to prevent them from resizing
        self.image_dimensions_label.setFixedWidth(150)
        self.mouse_position_label.setFixedWidth(150)
        self.view_dimensions_label.setFixedWidth(150)  # Set fixed width for view dimensions label

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)

        # Spin box for IoU threshold control
        self.iou_thresh_spinbox = QDoubleSpinBox()
        self.iou_thresh_spinbox.setRange(0.0, 1.0)  # Range is 0.0 to 1.0
        self.iou_thresh_spinbox.setSingleStep(0.05)  # Step size for the spinbox
        self.iou_thresh_spinbox.setValue(self.iou_thresh)
        self.iou_thresh_spinbox.valueChanged.connect(self.update_iou_thresh)

        # Spin box for Uncertainty threshold control
        self.uncertainty_thresh_spinbox = QDoubleSpinBox()
        self.uncertainty_thresh_spinbox.setRange(0.0, 1.0)  # Range is 0.0 to 1.0
        self.uncertainty_thresh_spinbox.setSingleStep(0.05)  # Step size for the spinbox
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
        self.status_bar_layout.addWidget(self.view_dimensions_label)  # Add view dimensions label to status bar layout
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("Transparency:"))
        self.status_bar_layout.addWidget(self.transparency_slider)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(QLabel("IoU Threshold:"))
        self.status_bar_layout.addWidget(self.iou_thresh_spinbox)
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

    def toggle_tool(self, state):
        action = self.sender()
        if action == self.select_tool_action:
            if state:
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("select")
            else:
                self.toolChanged.emit(None)
        elif action == self.patch_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("patch")
            else:
                self.toolChanged.emit(None)
        elif action == self.rectangle_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("rectangle")
            else:
                self.toolChanged.emit(None)
        elif action == self.polygon_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.toolChanged.emit("polygon")
            else:
                self.toolChanged.emit(None)
        elif action == self.sam_tool_action:
            if not self.sam_deploy_model_dialog.loaded_model:
                self.sam_tool_action.setChecked(False)
                QMessageBox.warning(self, "SAM Deploy Model", "You must deploy a model before using the SAM tool.")
                return
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.toolChanged.emit("sam")
            else:
                self.toolChanged.emit(None)

    def untoggle_all_tools(self):
        self.select_tool_action.setChecked(False)
        self.patch_tool_action.setChecked(False)
        self.rectangle_tool_action.setChecked(False)
        self.polygon_tool_action.setChecked(False)
        self.sam_tool_action.setChecked(False)
        self.toolChanged.emit(None)

    def handle_tool_changed(self, tool):
        if tool == "select":
            self.select_tool_action.setChecked(True)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "patch":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(True)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "rectangle":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(True)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
        elif tool == "polygon":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(True)
            self.sam_tool_action.setChecked(False)
        elif tool == "sam":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(True)
        else:
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)

    def toggle_device(self):
        dialog = DeviceSelectionDialog(self.devices, self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_devices = dialog.selected_devices

            if not self.selected_devices:
                return

            if len(self.selected_devices) == 1:
                self.device = self.selected_devices[0]
            else:
                cuda_devices = [device for device in self.selected_devices if device.startswith('cuda')]
                if cuda_devices:
                    self.device = f"cuda:{','.join(cuda_devices)}"
                else:
                    self.device = self.selected_devices[0]  # Default to the first selected device if no CUDA devices

            # Update the icon and tooltip
            if self.device.startswith('cuda'):
                if len(self.selected_devices) == 1:
                    device_icon = QIcon(self.rabbit_icon_path) if self.device == 'cuda:0' else QIcon(
                        self.rocket_icon_path)
                    device_tooltip = self.device
                else:
                    device_icon = QIcon(self.rocket_icon_path)  # Use a different icon for multiple devices
                    device_tooltip = f"Multiple CUDA Devices: {', '.join(self.selected_devices)}"
            elif self.device == 'mps':
                device_icon = QIcon(self.apple_icon_path)
                device_tooltip = 'mps'
            else:
                device_icon = QIcon(self.turtle_icon_path)
                device_tooltip = 'cpu'

            self.device_tool_action.setIcon(device_icon)
            self.device_tool_action.setToolTip(device_tooltip)

    def handle_image_changed(self):
        if self.annotation_window.selected_tool == 'sam':
            self.annotation_window.tools['sam'].cancel_working_area()

    def update_image_dimensions(self, width, height):
        self.image_dimensions_label.setText(f"Image: {height} x {width}")

    def update_mouse_position(self, x, y):
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")

    def update_view_dimensions(self, original_width, original_height):
        # Current extent (view)
        extent = self.annotation_window.viewportToScene()

        top = round(extent.top())
        left = round(extent.left())
        width = round(extent.width())
        height = round(extent.height())

        bottom = top + height
        right = left + width

        # If the current extent includes areas outside the
        # original image, reduce it to be only the original image
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if bottom > original_height:
            bottom = original_height
        if right > original_width:
            right = original_width

        width = right - left
        height = bottom - top

        self.view_dimensions_label.setText(f"View: {height} x {width}")

    def get_transparency_value(self):
        return self.transparency_slider.value()

    def update_label_transparency(self, value):
        self.label_window.set_label_transparency(value)
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

    def get_iou_thresh(self):
        return self.iou_thresh

    def update_iou_thresh(self, value):
        if self.iou_thresh != value:
            self.iou_thresh = value
            self.iou_thresh_spinbox.setValue(value)
            self.iouChanged.emit(value)

    def open_import_images_dialog(self):
        self.untoggle_all_tools()
        self.io_dialog.import_images()

    def open_patch_annotation_sampling_dialog(self):

        if not self.image_window.image_paths:
            # Check if there are any images in the project
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Proceed to open the dialog if images are loaded
            self.untoggle_all_tools()
            self.patch_annotation_sampling_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

        self.patch_annotation_sampling_dialog = None
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)


    def open_import_dataset_dialog(self):
        try:
            self.untoggle_all_tools()
            self.import_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_export_dataset_dialog(self):
        # Check if there are loaded images
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Export Dataset",
                                "No images are present in the project.")
            return

        # Check if there are annotations
        if not len(self.annotation_window.annotations_dict):
            QMessageBox.warning(self,
                                "Export Dataset",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_merge_datasets_dialog(self):
        try:
            self.untoggle_all_tools()
            self.merge_datasets_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not any(list(self.deploy_model_dialog.loaded_models.values())):
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_sam_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")


class ClickableAction(QAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.trigger()
        super().mousePressEvent(event)


class DeviceSelectionDialog(QDialog):
    def __init__(self, devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Device")
        self.devices = devices
        self.selected_devices = []

        layout = QVBoxLayout()

        self.device_list = QListWidget()
        self.device_list.addItems(self.devices)
        self.device_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.device_list)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def accept(self):
        self.selected_devices = [item.text() for item in self.device_list.selectedItems()]
        if self.validate_selection():
            super().accept()
        else:
            QMessageBox.warning(self, "Invalid Selection", "Cannot mix CUDA devices with CPU or MPS.")

    def validate_selection(self):
        cuda_selected = any(device.startswith('cuda') for device in self.selected_devices)
        cpu_selected = 'cpu' in self.selected_devices
        mps_selected = 'mps' in self.selected_devices

        if cuda_selected and (cpu_selected or mps_selected):
            return False
        return True