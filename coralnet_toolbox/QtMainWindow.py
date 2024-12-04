import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import re

from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QSize, QPoint
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtWidgets import (QListWidget, QCheckBox, QFrame)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy, 
                             QMessageBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                             QSpinBox, QSlider, QDialog, QPushButton, QToolButton,
                             QGroupBox)

from coralnet_toolbox.QtAnnotationWindow import AnnotationWindow
from coralnet_toolbox.QtConfidenceWindow import ConfidenceWindow
from coralnet_toolbox.QtImageWindow import ImageWindow
from coralnet_toolbox.QtLabelWindow import LabelWindow
from coralnet_toolbox.QtPatchSampling import PatchSamplingDialog
from coralnet_toolbox.QtEventFilter import GlobalEventFilter

from coralnet_toolbox.IO import (
    ImportImages,
    ImportLabels, 
    ImportAnnotations,
    ImportCoralNetAnnotations,
    ImportViscoreAnnotations,
    ImportTagLabAnnotations,
    ExportLabels,
    ExportAnnotations, 
    ExportCoralNetAnnotations,
    ExportViscoreAnnotations,
    ExportTagLabAnnotations
)

from coralnet_toolbox.MachineLearning import (
    TrainClassify as ClassifyTrainModelDialog,
    TrainDetect as DetectTrainModelDialog,  
    TrainSegment as SegmentTrainModelDialog,
    DeployClassify as ClassifyDeployModelDialog,
    DeployDetect as DetectDeployModelDialog,
    DeploySegment as SegmentDeployModelDialog,
    BatchClassify as ClassifyBatchInferenceDialog,
    BatchDetect as DetectBatchInferenceDialog,
    BatchSegment as SegmentBatchInferenceDialog,
    ImportDetect as DetectImportDatasetDialog,
    ImportSegment as SegmentImportDatasetDialog,
    ExportClassify as ClassifyExportDatasetDialog,
    ExportDetect as DetectExportDatasetDialog,
    ExportSegment as SegmentExportDatasetDialog,
    EvalClassify as ClassifyEvaluateModelDialog,
    EvalDetect as DetectEvaluateModelDialog,
    EvalSegment as SegmentEvaluateModelDialog,
    MergeClassify as ClassifyMergeDatasetsDialog,
    Optimize as OptimizeModelDialog
)

from coralnet_toolbox.SAM import (
    DeployPredictorDialog as SAMDeployPredictorDialog,
    DeployGeneratorDialog as SAMDeployGeneratorDialog,
    BatchInferenceDialog as SAMBatchInferenceDialog
)

from coralnet_toolbox.AutoDistill import (
    DeployModelDialog as AutoDistillDeployModelDialog,
    BatchInferenceDialog as AutoDistillBatchInferenceDialog
)

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.utilities import get_available_device


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state
    uncertaintyChanged = pyqtSignal(float)  # Signal to emit the current uncertainty threshold
    iouChanged = pyqtSignal(float)  # Signal to emit the current IoU threshold
    areaChanged = pyqtSignal(float, float)  # Signal to emit the current area threshold

    def __init__(self):
        super().__init__()
        
        # Define icons
        self.coral_icon = get_icon("coral.png")
        self.select_icon = get_icon("select.png")  
        self.patch_icon = get_icon("patch.png")
        self.rectangle_icon = get_icon("rectangle.png")
        self.polygon_icon = get_icon("polygon.png")
        self.sam_icon = get_icon("sam.png")
        self.slicer_icon = get_icon("slicer.png")
        self.turtle_icon = get_icon("turtle.png")
        self.rabbit_icon = get_icon("rabbit.png")
        self.rocket_icon = get_icon("rocket.png")
        self.apple_icon = get_icon("apple.png")
        self.transparent_icon = get_icon("transparent.png")
        self.opaque_icon = get_icon("opaque.png")
        self.parameters_icon = get_icon("parameters.png")

        # Set the title and icon for the main window
        self.setWindowTitle("CoralNet-Toolbox")
        self.setWindowIcon(self.coral_icon)

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

        self.import_images = ImportImages(self)
        self.import_labels = ImportLabels(self)
        self.import_annotations = ImportAnnotations(self)
        self.import_coralnet_annotations = ImportCoralNetAnnotations(self)
        self.import_viscore_annotations = ImportViscoreAnnotations(self)
        self.import_taglab_annotations = ImportTagLabAnnotations(self)
        self.export_labels = ExportLabels(self)
        self.export_annotations = ExportAnnotations(self)
        self.export_coralnet_annotations = ExportCoralNetAnnotations(self)
        self.export_viscore_annotations = ExportViscoreAnnotations(self)
        self.export_taglab_annotations = ExportTagLabAnnotations(self)

        # Set the default uncertainty threshold and IoU threshold
        self.iou_thresh = 0.20
        self.uncertainty_thresh = 0.30
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.40

        # Create dialogs
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)
        self.detect_import_dataset_dialog = DetectImportDatasetDialog(self)
        self.segment_import_dataset_dialog = SegmentImportDatasetDialog(self)
        self.classify_export_dataset_dialog = ClassifyExportDatasetDialog(self)
        self.detect_export_dataset_dialog = DetectExportDatasetDialog(self)
        self.segment_export_dataset_dialog = SegmentExportDatasetDialog(self)
        self.classify_merge_datasets_dialog = ClassifyMergeDatasetsDialog(self)
        self.classify_train_model_dialog = ClassifyTrainModelDialog(self)
        self.detect_train_model_dialog = DetectTrainModelDialog(self)
        self.segment_train_model_dialog = SegmentTrainModelDialog(self)
        self.classify_evaluate_model_dialog = ClassifyEvaluateModelDialog(self)
        self.detect_evaluate_model_dialog = DetectEvaluateModelDialog(self)
        self.segment_evaluate_model_dialog = SegmentEvaluateModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.classify_deploy_model_dialog = ClassifyDeployModelDialog(self)
        self.detect_deploy_model_dialog = DetectDeployModelDialog(self)
        self.segment_deploy_model_dialog = SegmentDeployModelDialog(self)
        self.classify_batch_inference_dialog = ClassifyBatchInferenceDialog(self)
        self.detect_batch_inference_dialog = DetectBatchInferenceDialog(self)
        self.segment_batch_inference_dialog = SegmentBatchInferenceDialog(self)
        self.sam_deploy_model_dialog = SAMDeployPredictorDialog(self)  # TODO
        self.sam_deploy_generator_dialog = SAMDeployGeneratorDialog(self)
        self.sam_batch_inference_dialog = SAMBatchInferenceDialog(self)
        self.auto_distill_deploy_model_dialog = AutoDistillDeployModelDialog(self)
        self.auto_distill_batch_inference_dialog = AutoDistillBatchInferenceDialog(self)

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

        # Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        # ----------------------------------------
        # Create the menu bar
        # ----------------------------------------
        self.menu_bar = self.menuBar()

        # Import menu
        self.import_menu = self.menu_bar.addMenu("Import")

        # Raster submenu
        self.import_rasters_menu = self.import_menu.addMenu("Rasters")

        # Import Images
        self.import_images_action = QAction("Images", self)
        self.import_images_action.triggered.connect(self.import_images.import_images)
        self.import_rasters_menu.addAction(self.import_images_action)

        # Labels submenu
        self.import_labels_menu = self.import_menu.addMenu("Labels")

        # Import Labels
        self.import_labels_action = QAction("Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.import_labels.import_labels)
        self.import_labels_menu.addAction(self.import_labels_action)

        # Annotations submenu
        self.import_annotations_menu = self.import_menu.addMenu("Annotations")

        # Import Annotations
        self.import_annotations_action = QAction("Annotations (JSON)", self)
        self.import_annotations_action.triggered.connect(self.import_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_annotations_action)

        # Import CoralNet Annotations
        self.import_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.import_coralnet_annotations_action.triggered.connect(self.import_coralnet_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_coralnet_annotations_action)

        # Import Viscore Annotations
        self.import_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.import_viscore_annotations_action.triggered.connect(self.import_viscore_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_viscore_annotations_action)

        # Import TagLab Annotations
        self.import_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.import_taglab_annotations_action.triggered.connect(self.import_taglab_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_taglab_annotations_action)

        # Dataset submenu
        self.import_dataset_menu = self.import_menu.addMenu("Dataset")
        
        # Import Detection Dataset submenu
        self.import_detect_dataset_action = QAction("Detect", self)
        self.import_detect_dataset_action.triggered.connect(self.detect_import_dataset_dialog.exec_)
        self.import_dataset_menu.addAction(self.import_detect_dataset_action)

        # Import Segmentation Dataset submenu
        self.import_segment_dataset_action = QAction("Segment", self)
        self.import_segment_dataset_action.triggered.connect(self.segment_import_dataset_dialog.exec_)
        self.import_dataset_menu.addAction(self.import_segment_dataset_action)

        # Export menu
        self.export_menu = self.menu_bar.addMenu("Export")

        # Labels submenu
        self.export_labels_menu = self.export_menu.addMenu("Labels")

        # Export Labels
        self.export_labels_action = QAction("Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.export_labels.export_labels)
        self.export_labels_menu.addAction(self.export_labels_action)

        # Annotations submenu
        self.export_annotations_menu = self.export_menu.addMenu("Annotations")

        # Export Annotations
        self.export_annotations_action = QAction("Annotations (JSON)", self)
        self.export_annotations_action.triggered.connect(self.export_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_annotations_action)

        # Export CoralNet Annotations
        self.export_coralnet_annotations_action = QAction("CoralNet (CSV)", self)
        self.export_coralnet_annotations_action.triggered.connect(self.export_coralnet_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_coralnet_annotations_action)

        # Export Viscore Annotations 
        self.export_viscore_annotations_action = QAction("Viscore (CSV)", self)
        self.export_viscore_annotations_action.triggered.connect(self.export_viscore_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_viscore_annotations_action)

        # Export TagLab Annotations
        self.export_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.export_taglab_annotations_action.triggered.connect(self.export_taglab_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_taglab_annotations_action)

        # Dataset submenu
        self.export_dataset_menu = self.export_menu.addMenu("Dataset")
        
        # Export Classification Dataset 
        self.export_classify_dataset_action = QAction("Classify", self)
        self.export_classify_dataset_action.triggered.connect(self.open_classify_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_classify_dataset_action)
        
        # Export Detection Dataset 
        self.export_detect_dataset_action = QAction("Detect", self)
        self.export_detect_dataset_action.triggered.connect(self.open_detect_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_detect_dataset_action)
        
        # Export Segmentation Dataset 
        self.export_segment_dataset_action = QAction("Segment", self)
        self.export_segment_dataset_action.triggered.connect(self.open_segment_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_segment_dataset_action)

        # Sampling Annotations menu
        self.annotation_sampling_action = QAction("Sample", self)
        self.annotation_sampling_action.triggered.connect(self.open_patch_annotation_sampling_dialog)
        self.menu_bar.addAction(self.annotation_sampling_action)

        # CoralNet menu
        # self.coralnet_menu = self.menu_bar.addMenu("CoralNet")

        # self.coralnet_authenticate_action = QAction("Authenticate", self)
        # self.coralnet_authenticate_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_authenticate_action)

        # self.coralnet_upload_action = QAction("Upload", self)
        # self.coralnet_upload_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_upload_action)

        # self.coralnet_download_action = QAction("Download", self)
        # self.coralnet_download_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_download_action)

        # self.coralnet_model_api_action = QAction("Model API", self)
        # self.coralnet_model_api_action.triggered.connect(
        #     lambda: QMessageBox.information(self, "Placeholder", "This is not yet implemented."))
        # self.coralnet_menu.addAction(self.coralnet_model_api_action)

        # Ultralytics menu
        self.ml_menu = self.menu_bar.addMenu("Ultralytics")

        # Merge Datasets submenu
        self.ml_merge_datasets_menu = self.ml_menu.addMenu("Merge Datasets")
        
        # Merge Classification Datasets
        self.ml_classify_merge_datasets_action = QAction("Classify", self)
        self.ml_classify_merge_datasets_action.triggered.connect(self.open_classify_merge_datasets_dialog)
        self.ml_merge_datasets_menu.addAction(self.ml_classify_merge_datasets_action)

        # Train Model submenu
        self.ml_train_model_menu = self.ml_menu.addMenu("Train Model")

        # Train Classification Model
        self.ml_classify_train_model_action = QAction("Classify", self)
        self.ml_classify_train_model_action.triggered.connect(self.open_classify_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_classify_train_model_action)

        # Train Detection Model
        self.ml_detect_train_model_action = QAction("Detect", self)
        self.ml_detect_train_model_action.triggered.connect(self.open_detect_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_detect_train_model_action)

        # Train Segmentation Model
        self.ml_segment_train_model_action = QAction("Segment", self)
        self.ml_segment_train_model_action.triggered.connect(self.open_segment_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_segment_train_model_action)

        # Evaluate Model submenu 
        self.ml_evaluate_model_menu = self.ml_menu.addMenu("Evaluate Model")
        
        # Evaluate Classification Model
        self.ml_classify_evaluate_model_action = QAction("Classify", self)
        self.ml_classify_evaluate_model_action.triggered.connect(self.open_classify_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_classify_evaluate_model_action)
        
        # Evaluate Detection Model
        self.ml_detect_evaluate_model_action = QAction("Detect", self)
        self.ml_detect_evaluate_model_action.triggered.connect(self.open_detect_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_detect_evaluate_model_action)
        
        # Evaluate Segmentation Model
        self.ml_segment_evaluate_model_action = QAction("Segment", self)
        self.ml_segment_evaluate_model_action.triggered.connect(self.open_segment_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_segment_evaluate_model_action)
        
        # Optimize Model 
        self.ml_optimize_model_action = QAction("Optimize Model", self)
        self.ml_optimize_model_action.triggered.connect(self.open_optimize_model_dialog)
        self.ml_menu.addAction(self.ml_optimize_model_action)

        # Deploy Model submenu
        self.ml_deploy_model_menu = self.ml_menu.addMenu("Deploy Model")

        # Deploy Classification Model
        self.ml_classify_deploy_model_action = QAction("Classify", self)
        self.ml_classify_deploy_model_action.triggered.connect(self.open_classify_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_classify_deploy_model_action)

        # Deploy Detection Model
        self.ml_detect_deploy_model_action = QAction("Detect", self)
        self.ml_detect_deploy_model_action.triggered.connect(self.open_detect_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_detect_deploy_model_action)

        # Deploy Segmentation Model
        self.ml_segment_deploy_model_action = QAction("Segment", self)
        self.ml_segment_deploy_model_action.triggered.connect(self.open_segment_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_segment_deploy_model_action)

        # Batch Inference submenu
        self.ml_batch_inference_menu = self.ml_menu.addMenu("Batch Inference")

        # Batch Inference Classification
        self.ml_classify_batch_inference_action = QAction("Classify", self)
        self.ml_classify_batch_inference_action.triggered.connect(self.open_classify_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_classify_batch_inference_action)

        # Batch Inference Detection
        self.ml_detect_batch_inference_action = QAction("Detect", self)
        self.ml_detect_batch_inference_action.triggered.connect(self.open_detect_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_detect_batch_inference_action)

        # Batch Inference Segmentation
        self.ml_segment_batch_inference_action = QAction("Segment", self)
        self.ml_segment_batch_inference_action.triggered.connect(self.open_segment_batch_inference_dialog)
        self.ml_batch_inference_menu.addAction(self.ml_segment_batch_inference_action)

        # SAM menu
        self.sam_menu = self.menu_bar.addMenu("SAM")
        
        # Deploy Model submenu
        self.sam_deploy_model_menu = self.sam_menu.addMenu("Deploy Model")
        
        # Deploy Predictor
        self.sam_deploy_model_action = QAction("Predictor", self)
        self.sam_deploy_model_action.triggered.connect(self.open_sam_deploy_model_dialog)
        self.sam_deploy_model_menu.addAction(self.sam_deploy_model_action)
        
        # Deploy Generator
        self.sam_deploy_generator_action = QAction("Generator", self)
        self.sam_deploy_generator_action.triggered.connect(self.open_sam_deploy_generator_dialog)
        self.sam_deploy_model_menu.addAction(self.sam_deploy_generator_action)
        
        # Batch Inference
        self.sam_batch_inference_action = QAction("Batch Inference", self)
        self.sam_batch_inference_action.triggered.connect(self.open_sam_batch_inference_dialog)
        self.sam_menu.addAction(self.sam_batch_inference_action)

        # Auto Distill menu
        self.auto_distill_menu = self.menu_bar.addMenu("AutoDistill")

        # Deploy Model
        self.auto_distill_deploy_model_action = QAction("Deploy Model", self)
        self.auto_distill_deploy_model_action.triggered.connect(self.open_auto_distill_deploy_model_dialog)
        self.auto_distill_menu.addAction(self.auto_distill_deploy_model_action)
        
        # Batch Inference
        self.auto_distill_batch_inference_action = QAction("Batch Inference", self)
        self.auto_distill_batch_inference_action.triggered.connect(self.open_auto_distill_batch_inference_dialog)
        self.auto_distill_menu.addAction(self.auto_distill_batch_inference_action)

        # ----------------------------------------
        # Create and add the toolbar
        # ----------------------------------------
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setOrientation(Qt.Vertical)
        self.toolbar.setFixedWidth(40)
        self.toolbar.setMovable(False)  # Lock the toolbar in place
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        # Define spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        spacer.setFixedHeight(10)  # Set a fixed height for the spacer
        
        # Define line separator
        separator = QWidget()
        separator.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        separator.setFixedHeight(1)  # Set a fixed height for the line separator
        
        # Add a spacer before the first tool with a fixed height
        self.toolbar.addWidget(spacer)

        # Add tools here with icons
        self.select_tool_action = QAction(self.select_icon, "Select", self)
        self.select_tool_action.setCheckable(True)
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)
        
        self.toolbar.addSeparator()
        
        self.patch_tool_action = QAction(self.patch_icon, "Patch", self)
        self.patch_tool_action.setCheckable(True)
        self.patch_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.patch_tool_action)

        self.rectangle_tool_action = QAction(self.rectangle_icon, "Rectangle", self)
        self.rectangle_tool_action.setCheckable(True)
        self.rectangle_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.rectangle_tool_action)

        self.polygon_tool_action = QAction(self.polygon_icon, "Polygon", self)
        self.polygon_tool_action.setCheckable(True)
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)
        
        self.toolbar.addSeparator()
        
        self.sam_tool_action = QAction(self.sam_icon, "SAM", self)
        self.sam_tool_action.setCheckable(True)
        self.sam_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.sam_tool_action)
        
        self.toolbar.addSeparator()
        
        self.slicer_tool_action = QAction(self.slicer_icon, "Slicer", self)
        self.slicer_tool_action.setCheckable(False)
        self.slicer_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.slicer_tool_action)
        
        self.toolbar.addSeparator()

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
                device_icon = self.rabbit_icon
            else:
                device_icon = self.rocket_icon
            device_tooltip = self.device
        elif self.device == 'mps':
            device_icon = self.apple_icon
            device_tooltip = 'mps'
        else:
            device_icon = self.turtle_icon
            device_tooltip = 'cpu'

        # Create the device action with the appropriate icon
        device_action = ClickableAction(device_icon, "", self)  # Empty string for the text
        self.device_tool_action = device_action
        self.device_tool_action.setCheckable(False)
        # Set the tooltip to show the value of self.device
        self.device_tool_action.setToolTip(device_tooltip)
        self.device_tool_action.triggered.connect(self.toggle_device)
        self.toolbar.addAction(self.device_tool_action)
        
        # ----------------------------------------
        # Create and add the status bar
        # ----------------------------------------
        self.status_bar_layout = QHBoxLayout()

        # Labels for image dimensions and mouse position
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.view_dimensions_label = QLabel("View: 0 x 0")  # Add QLabel for view dimensions

        # Set fixed width for labels to prevent them from resizing
        self.image_dimensions_label.setFixedWidth(150)
        self.mouse_position_label.setFixedWidth(150)
        self.view_dimensions_label.setFixedWidth(150)  # Set fixed width for view dimensions label

        # Transparency slider with icons
        transparency_layout = QHBoxLayout()
        transparent_icon = QLabel()
        transparent_icon.setPixmap(get_icon("transparent.png").pixmap(QSize(16, 16)))
        transparent_icon.setToolTip("Transparent")
        
        # Slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 128)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)
        
        # Right icon (opaque)
        opaque_icon = QLabel()
        opaque_icon.setPixmap(get_icon("opaque.png").pixmap(QSize(16, 16)))
        opaque_icon.setToolTip("Opaque")
        
        # Add a checkbox labeled "All" next to the transparency slider
        self.all_labels_checkbox = QCheckBox("")
        self.all_labels_checkbox.setCheckState(Qt.Checked)
        self.all_labels_checkbox.stateChanged.connect(self.update_all_labels_transparency)
        
        # Add widgets to the transparency layout
        transparency_layout.addWidget(transparent_icon)
        transparency_layout.addWidget(self.transparency_slider)
        transparency_layout.addWidget(opaque_icon)
        transparency_layout.addWidget(self.all_labels_checkbox)
        
        # Create widget to hold the layout
        self.transparency_widget = QWidget()
        self.transparency_widget.setLayout(transparency_layout)
        
        # --------------------------------------------------
        # Create collapsible Parameters section
        # --------------------------------------------------
        self.parameters_section = CollapsibleSection("Parameters")
        
        # Patch Annotation Size
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(5000)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)
        annotation_size_layout = QHBoxLayout()
        annotation_size_layout.addWidget(self.annotation_size_spinbox)
        annotation_size_widget = QWidget()
        annotation_size_widget.setLayout(annotation_size_layout)
        self.parameters_section.add_widget(annotation_size_widget, "Patch Size")
        
        # Uncertainty threshold
        self.uncertainty_thresh_slider = QSlider(Qt.Horizontal)
        self.uncertainty_thresh_slider.setRange(0, 100)
        self.uncertainty_thresh_slider.setValue(int(self.uncertainty_thresh * 100))
        self.uncertainty_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.uncertainty_thresh_slider.setTickInterval(10)
        self.uncertainty_value_label = QLabel(f"{self.uncertainty_thresh:.2f}")
        self.uncertainty_thresh_slider.valueChanged.connect(self.update_uncertainty_label)
        uncertainty_layout = QHBoxLayout()
        uncertainty_layout.addWidget(self.uncertainty_thresh_slider)
        uncertainty_layout.addWidget(self.uncertainty_value_label)
        uncertainty_widget = QWidget()
        uncertainty_widget.setLayout(uncertainty_layout)
        self.parameters_section.add_widget(uncertainty_widget, "Uncertainty Threshold")

        # IoU threshold
        self.iou_thresh_slider = QSlider(Qt.Horizontal)
        self.iou_thresh_slider.setRange(0, 100)
        self.iou_thresh_slider.setValue(int(self.iou_thresh * 100))
        self.iou_thresh_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_thresh_slider.setTickInterval(10)
        self.iou_value_label = QLabel(f"{self.iou_thresh:.2f}")
        self.iou_thresh_slider.valueChanged.connect(self.update_iou_label)
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(self.iou_thresh_slider)
        iou_layout.addWidget(self.iou_value_label)
        iou_widget = QWidget()
        iou_widget.setLayout(iou_layout)
        self.parameters_section.add_widget(iou_widget, "IoU Threshold")
        
        # Area threshold controls  
        min_val = self.area_thresh_min
        max_val = self.area_thresh_max
        self.area_threshold_slider = QRangeSlider(Qt.Horizontal)
        self.area_threshold_slider.setMinimum(0)
        self.area_threshold_slider.setMaximum(100)
        self.area_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_slider.setTickInterval(10)
        self.area_threshold_slider.setValue((int(min_val * 100), int(max_val * 100)))
        self.area_threshold_label = QLabel(f"{min_val:.2f} - {max_val:.2f}")
        self.area_threshold_slider.valueChanged.connect(self.update_area_label)
        area_thresh_layout = QVBoxLayout()
        area_thresh_layout.addWidget(self.area_threshold_slider) 
        area_thresh_layout.addWidget(self.area_threshold_label)
        area_thresh_widget = QWidget()
        area_thresh_widget.setLayout(area_thresh_layout)
        self.parameters_section.add_widget(area_thresh_widget, "Area Threshold")
        
        # Add widgets to status bar layout
        self.status_bar_layout.addWidget(self.image_dimensions_label)
        self.status_bar_layout.addWidget(self.mouse_position_label)
        self.status_bar_layout.addWidget(self.view_dimensions_label)
        self.status_bar_layout.addWidget(self.transparency_widget)
        self.status_bar_layout.addStretch()
        self.status_bar_layout.addWidget(self.parameters_section)

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
                QMessageBox.warning(self, 
                                    "SAM Deploy Predictor", 
                                    "You must deploy a Predictor before using the SAM tool.")
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

            # Convert the string to multi-CUDA format: "cuda:0,1,2"
            if len(self.selected_devices) == 1:
                self.device = self.selected_devices[0]
            else:
                # Get only the numerical values for cuda
                cuda_devices = [re.search(r'\d+', device).group() for device in self.selected_devices]
                self.device = f"{','.join(cuda_devices)} "

            # Update the icon and tooltip
            if self.device.startswith('cuda'):
                if len(self.selected_devices) == 1:
                    if self.device == 'cuda:0':
                        device_icon = self.rabbit_icon
                    else:
                        # Use a different icon for other CUDA devices
                        device_icon = self.rabbit_icon
                    device_tooltip = self.device
                else:
                    # Use a different icon for multiple devices
                    device_icon = self.rocket_icon
                    device_tooltip = f"Multiple CUDA Devices: {self.selected_devices}"

            elif self.device == 'mps':
                device_icon = self.apple_icon
                device_tooltip = 'mps'
            else:
                device_icon = self.turtle_icon
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

    def update_transparency_slider(self, transparency):
        self.transparency_slider.setValue(transparency)
        
    def update_label_transparency(self, value):
        if self.all_labels_checkbox.isChecked():
            self.label_window.set_all_labels_transparency(value)
        else:
            self.label_window.set_label_transparency(value)
        self.update_transparency_slider(value)

    def update_all_labels_transparency(self, state):
        if state == Qt.Checked:
            self.label_window.set_all_labels_transparency(self.transparency_slider.value())
        else:
            self.label_window.set_label_transparency(self.transparency_slider.value())

    def get_uncertainty_thresh(self):
        """Get the current uncertainty threshold value"""
        return self.uncertainty_thresh

    def update_uncertainty_thresh(self, value):
        """Update the uncertainty threshold value"""
        if self.uncertainty_thresh != value:
            self.uncertainty_thresh = value
            self.uncertainty_thresh_slider.setValue(int(value * 100))  # Convert to slider range (0-100)
            self.uncertaintyChanged.emit(value)
            
    def update_uncertainty_label(self, value):
        """Update uncertainty threshold label when slider value changes"""
        self.uncertainty_thresh = value / 100.0  # Convert from 0-100 to 0-1
        self.uncertainty_value_label.setText(f"{self.uncertainty_thresh:.2f}")

    def get_iou_thresh(self):
        """Get the current IoU threshold value"""
        return self.iou_thresh

    def update_iou_thresh(self, value):
        """Update the IoU threshold value"""
        if self.iou_thresh != value:
            self.iou_thresh = value
            self.iou_thresh_slider.setValue(int(value * 100))  # Convert to slider range (0-100)
            self.iouChanged.emit(value)
            
    def update_iou_label(self, value):
        """Update IoU threshold label when slider value changes"""
        self.iou_thresh = value / 100.0  # Convert from 0-100 to 0-1
        self.iou_value_label.setText(f"{self.iou_thresh:.2f}")
        
    def get_area_thresh(self):
        """Get the current area threshold values"""
        return self.area_thresh_min, self.area_thresh_max
    
    def get_area_thresh_min(self):
        """Get the current minimum area threshold value"""
        return self.area_thresh_min
    
    def get_area_thresh_max(self):
        """Get the current maximum area threshold value"""
        return self.area_thresh_max
    
    def update_area_thresh(self, min_val, max_val):
        """Update the area threshold values"""
        if self.area_thresh_min != min_val or self.area_thresh_max != max_val:
            self.area_thresh_min = min_val
            self.area_thresh_max = max_val
            self.area_threshold_slider.setValue((int(min_val * 100), int(max_val * 100)))
            self.areaChanged.emit(min_val, max_val)
            
    def update_area_label(self, value):
        """Handle changes to area threshold range slider"""
        min_val, max_val = self.area_threshold_slider.value()  # Returns tuple of values
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")

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
            
    def open_classify_export_dataset_dialog(self):
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
            self.classify_export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_detect_export_dataset_dialog(self):
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
            self.detect_export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_segment_export_dataset_dialog(self):
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
            self.segment_export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_merge_datasets_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_merge_datasets_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.detect_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_train_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.segment_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.classify_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_detect_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.detect_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_segment_evaluate_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.segment_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        try:
            self.untoggle_all_tools()
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Classify Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.classify_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Detect Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.detect_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Segment Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.segment_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.classify_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return
        
        if not self.annotation_window.annotations_dict:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.classify_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.detect_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.detect_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "No images are present in the project.")
            return

        if not self.segment_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.segment_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_sam_deploy_model_dialog(self):  # TODO 
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Predictor",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_model_dialog.exec_()  
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_sam_deploy_generator_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Generator",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_generator_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_sam_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "SAM Batch Inference",
                                "No images are present in the project.")
            return

        if not self.sam_deploy_generator_dialog.loaded_model:
            QMessageBox.warning(self,
                                "SAM Batch Inference",
                                "Please deploy a generator before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_auto_distill_deploy_model_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "AutoDistill Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.auto_distill_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_auto_distill_batch_inference_dialog(self):
        if not self.image_window.image_paths:
            QMessageBox.warning(self,
                                "AutoDistill Batch Inference",
                                "No images are present in the project.")
            return

        if not self.auto_distill_deploy_model_dialog.loaded_model:
            QMessageBox.warning(self,
                                "AutoDistill Batch Inference",
                                "Please deploy a model before running batch inference.")
            return

        try:
            self.untoggle_all_tools()
            self.auto_distill_batch_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        # Create the action
        self.toggle_action = QAction(QIcon(get_icon('parameters.png')), title, self)
        self.toggle_action.setCheckable(True)
        self.toggle_action.triggered.connect(self.toggle_content)

        # Header button using the action
        self.toggle_button = QToolButton()
        self.toggle_button.setDefaultAction(self.toggle_action)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setAutoRaise(True)  # Gives a flat appearance until clicked

        # Popup frame
        self.popup = QFrame(self.window())
        self.popup.setWindowFlags(Qt.Popup)
        self.popup.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.popup.setLayout(QVBoxLayout())
        self.popup.layout().setContentsMargins(5, 5, 5, 5)
        self.popup.hide()

        # Add button to layout
        self.layout().addWidget(self.toggle_button)

    def toggle_content(self, checked):
        if checked:
            # Position popup below and to the left of the button
            pos = self.toggle_button.mapToGlobal(QPoint(0, 0))
            popup_width = self.popup.sizeHint().width()
            self.popup.move(pos.x() - popup_width + self.toggle_button.width(), 
                            pos.y() + self.toggle_button.height())
            self.popup.show()
        else:
            self.popup.hide()

    def add_widget(self, widget, title=None):
        group_box = QGroupBox()
        group_box.setTitle(title)
        group_box.setLayout(QVBoxLayout())
        group_box.layout().addWidget(widget)
        self.popup.layout().addWidget(group_box)

    def hideEvent(self, event):
        self.popup.hide()
        self.toggle_action.setChecked(False)
        super().hideEvent(event)
        

class DeviceSelectionDialog(QDialog):
    def __init__(self, devices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Device")
        self.devices = devices
        self.selected_devices = []

        layout = QVBoxLayout()

        self.device_list = QListWidget()
        self.device_list.addItems(self.devices)
        self.device_list.setSelectionMode(QListWidget.SingleSelection)  # Allow only single selection
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


class ClickableAction(QAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.trigger()
        super().mousePressEvent(event)