import warnings

import os
import re
import ctypes
import requests

from packaging import version

import numpy as np
import torch

from PyQt5 import sip
from PyQt5.QtGui import QIcon, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QSize, QPoint
from PyQt5.QtWidgets import (QListWidget, QCheckBox, QFrame, QComboBox)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy,
                             QMessageBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                             QSpinBox, QSlider, QDialog, QPushButton, QToolButton,
                             QGroupBox, QSpacerItem)

from coralnet_toolbox.QtEventFilter import GlobalEventFilter
from coralnet_toolbox.QtAnimationManager import AnimationManager
from coralnet_toolbox.QtSystemMonitor import SystemMonitor
from coralnet_toolbox.QtTimer import TimerGroupBox

from coralnet_toolbox.QtAnnotationWindow import AnnotationWindow
from coralnet_toolbox.QtConfidenceWindow import ConfidenceWindow
from coralnet_toolbox.QtImageWindow import ImageWindow
from coralnet_toolbox.QtLabelWindow import LabelWindow

from coralnet_toolbox.Explorer import ExplorerWindow

from coralnet_toolbox.QtPatchSampling import PatchSamplingDialog
from coralnet_toolbox.QtBatchInference import BatchInferenceDialog


from coralnet_toolbox.Tile import (
    TileClassifyDataset as ClassifyTileDatasetDialog,
    TileDetectDataset as DetectTileDatasetDialog,
    TileSegmentDataset as SegmentTileDatasetDialog,
    TileSemanticDataset as SemanticTileDatasetDialog,
    TileManager as TileManagerDialog,
)

# TODO update IO classes to have dialogs
from coralnet_toolbox.IO import (
    ImportImages,
    ImportFrames,
    ImportLabels,
    ImportCoralNetLabels,
    ImportTagLabLabels,
    ImportAnnotations,
    ImportCoralNetAnnotations,
    ImportViscoreAnnotations,
    ImportTagLabAnnotations,
    ImportSquidleAnnotations,
    ExportLabels,
    ExportTagLabLabels,
    ExportAnnotations,
    ExportMaskAnnotations,
    ExportGeoJSONAnnotations,
    ExportCoralNetAnnotations,
    ExportViscoreAnnotations,
    ExportTagLabAnnotations,
    OpenProject,
    SaveProject
)

from coralnet_toolbox.MachineLearning import (
    TuneClassify as ClassifyTuneDialog,
    TuneDetect as DetectTuneDialog,
    TuneSegment as SegmentTuneDialog,
    TuneSemantic as SemanticTuneDialog,
    TrainClassify as ClassifyTrainModelDialog,
    TrainDetect as DetectTrainModelDialog,
    TrainSegment as SegmentTrainModelDialog,
    TrainSemantic as SemanticTrainModelDialog,
    DeployClassify as ClassifyDeployModelDialog,
    DeployDetect as DetectDeployModelDialog,
    DeploySegment as SegmentDeployModelDialog,
    DeploySemantic as SemanticDeployModelDialog,
    VideoDetect as DetectVideoInferenceDialog,
    VideoSegment as SegmentVideoInferenceDialog,
    ImportDetect as DetectImportDatasetDialog,
    ImportSegment as SegmentImportDatasetDialog,
    ExportClassify as ClassifyExportDatasetDialog,
    ExportDetect as DetectExportDatasetDialog,
    ExportSegment as SegmentExportDatasetDialog,
    ExportSemantic as SemanticExportDatasetDialog,
    EvalClassify as ClassifyEvaluateModelDialog,
    EvalDetect as DetectEvaluateModelDialog,
    EvalSegment as SegmentEvaluateModelDialog,
    EvalSemantic as SemanticEvaluateModelDialog,
    MergeClassify as ClassifyMergeDatasetsDialog,
    Optimize as OptimizeModelDialog
)

from coralnet_toolbox.SAM import (
    DeployPredictorDialog as SAMDeployPredictorDialog,
    DeployGeneratorDialog as SAMDeployGeneratorDialog,
)

from coralnet_toolbox.SeeAnything import (
    TrainModelDialog as SeeAnythingTrainModelDialog,
    DeployPredictorDialog as SeeAnythingDeployPredictorDialog,
    DeployGeneratorDialog as SeeAnythingDeployGeneratorDialog,
)

from coralnet_toolbox.Transformers import (
    DeployModelDialog as TransformersDeployModelDialog,
)

from coralnet_toolbox.CoralNet import (
    AuthenticateDialog as CoralNetAuthenticateDialog,
    DownloadDialog as CoralNetDownloadDialog
)

from coralnet_toolbox.BreakTime import (
    SnakeGame,
    BreakoutGame,
    LightCycleGame
)

from coralnet_toolbox.utilities import convert_scale_units

from coralnet_toolbox.Icons import get_icon


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MainWindow(QMainWindow):
    toolChanged = pyqtSignal(str)  # Signal to emit the current tool state
    maxDetectionsChanged = pyqtSignal(int)  # Signal to emit the current max detections value
    uncertaintyChanged = pyqtSignal(float)  # Signal to emit the current uncertainty threshold
    iouChanged = pyqtSignal(float)  # Signal to emit the current IoU threshold
    areaChanged = pyqtSignal(float, float)  # Signal to emit the current area threshold

    def __init__(self, __version__):
        super().__init__()
        
        # Create the animation manager
        self.animation_manager = AnimationManager(self)
        self.animation_manager.start_timer(interval=50)  # 50ms interval
        
        # Get the process ID
        self.pid = os.getpid()

        # Define icons
        self.coralnet_icon = get_icon("coralnet.png")
        self.coral_icon = get_icon("coral.png")
        self.select_icon = get_icon("select.png")
        self.patch_icon = get_icon("patch.png")
        self.rectangle_icon = get_icon("rectangle.png")
        self.polygon_icon = get_icon("polygon.png")
        self.brush_icon = get_icon("brush.png")
        self.erase_icon = get_icon("erase.png")
        self.dropper_icon = get_icon("dropper.png")
        self.fill_icon = get_icon("fill.png")
        self.sam_icon = get_icon("wizard.png")
        self.see_anything_icon = get_icon("eye.png")
        self.tile_icon = get_icon("tile.png")
        self.workarea_icon = get_icon("workarea.png")
        self.scale_icon = get_icon("scale.png")
        self.turtle_icon = get_icon("turtle.png")
        self.rabbit_icon = get_icon("rabbit.png")
        self.rocket_icon = get_icon("rocket.png")
        self.apple_icon = get_icon("apple.png")
        self.transparent_icon = get_icon("transparent.png")
        self.opaque_icon = get_icon("opaque.png")
        self.z_icon = get_icon("z.png")
        self.dynamic_icon = get_icon("dynamic.png")
        self.parameters_icon = get_icon("parameters.png")
        self.system_monitor_icon = get_icon("system_monitor.png")
        self.add_icon = get_icon("add.png")
        self.remove_icon = get_icon("remove.png")
        self.edit_icon = get_icon("edit.png")
        self.lock_icon = get_icon("lock.png")
        self.unlock_icon = get_icon("unlock.png")
        self.home_icon = get_icon("home.png")

        # Set the version
        self.version = __version__

        # Project path
        self.current_project_path = ""

        # Update the project label
        self.update_project_label()

        # Set icon
        self.setWindowIcon(self.coralnet_icon)

        # Set window flags for resizing, minimize, maximize, and customizing
        self.setWindowFlags(Qt.Window |
                            Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint |
                            Qt.WindowTitleHint)
        
        # Store view dimensions in base unit (meters)
        self.scaled_view_width_m = 0.0
        self.scaled_view_height_m = 0.0

        # Set the default uncertainty threshold and IoU threshold
        self.max_detections = 500
        self.iou_thresh = 0.50
        self.uncertainty_thresh = 0.20
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.70

        # Set the default scale unit
        self.current_unit_scale = 'm'
        # Set the default z-channel unit
        self.current_unit_z = 'm'
        
        # Cache the raw z-value at current mouse position to enable re-conversion
        # when user changes z-unit dropdown, without needing to re-read from raster
        self.current_z_value = None

        # Store current mouse position for z-channel lookup
        self.current_mouse_x = 0
        self.current_mouse_y = 0

        # Create windows
        self.annotation_window = AnnotationWindow(self)
        self.image_window = ImageWindow(self)
        self.label_window = LabelWindow(self)
        self.confidence_window = ConfidenceWindow(self)
        
        self.explorer_window = None  # Initialized in open_explorer_window
        self.system_monitor = None  # Initialized in open_system_monitor

        # TODO update IO classes to have dialogs
        # Create dialogs (I/O)
        self.import_images = ImportImages(self)
        self.import_labels = ImportLabels(self)
        self.import_coralnet_labels = ImportCoralNetLabels(self)
        self.import_taglab_labels = ImportTagLabLabels(self)
        self.import_annotations = ImportAnnotations(self)
        self.import_coralnet_annotations = ImportCoralNetAnnotations(self)
        self.import_viscore_annotations_dialog = ImportViscoreAnnotations(self)
        self.import_taglab_annotations = ImportTagLabAnnotations(self)
        self.import_squidle_annotations = ImportSquidleAnnotations(self)
        self.export_labels = ExportLabels(self)
        self.export_taglab_labels = ExportTagLabLabels(self)
        self.export_annotations = ExportAnnotations(self)
        self.export_coralnet_annotations = ExportCoralNetAnnotations(self)
        self.export_viscore_annotations_dialog = ExportViscoreAnnotations(self)
        self.export_taglab_annotations = ExportTagLabAnnotations(self)
        self.export_mask_annotations_dialog = ExportMaskAnnotations(self)
        self.export_geojson_annotations_dialog = ExportGeoJSONAnnotations(self)
        self.import_frames_dialog = ImportFrames(self)
        self.open_project_dialog = OpenProject(self)
        self.save_project_dialog = SaveProject(self)

        # Create dialogs (Sample)
        self.patch_annotation_sampling_dialog = PatchSamplingDialog(self)

        # Create dialogs (CoralNet)
        self.coralnet_authenticate_dialog = CoralNetAuthenticateDialog(self)
        self.coralnet_download_dialog = CoralNetDownloadDialog(self)

        # Create dialogs (Machine Learning)
        self.detect_import_dataset_dialog = DetectImportDatasetDialog(self)
        self.segment_import_dataset_dialog = SegmentImportDatasetDialog(self)
        self.classify_export_dataset_dialog = ClassifyExportDatasetDialog(self)
        self.detect_export_dataset_dialog = DetectExportDatasetDialog(self)
        self.segment_export_dataset_dialog = SegmentExportDatasetDialog(self)
        self.semantic_export_dataset_dialog = SemanticExportDatasetDialog(self)
        self.classify_merge_datasets_dialog = ClassifyMergeDatasetsDialog(self)
        self.classify_tune_model_dialog = ClassifyTuneDialog(self)
        self.detect_tune_model_dialog = DetectTuneDialog(self)
        self.segment_tune_model_dialog = SegmentTuneDialog(self)
        self.semantic_tune_model_dialog = SemanticTuneDialog(self)
        self.classify_train_model_dialog = ClassifyTrainModelDialog(self)
        self.detect_train_model_dialog = DetectTrainModelDialog(self)
        self.segment_train_model_dialog = SegmentTrainModelDialog(self)
        self.semantic_train_model_dialog = SemanticTrainModelDialog(self)
        self.classify_evaluate_model_dialog = ClassifyEvaluateModelDialog(self)
        self.detect_evaluate_model_dialog = DetectEvaluateModelDialog(self)
        self.segment_evaluate_model_dialog = SegmentEvaluateModelDialog(self)
        self.semantic_evaluate_model_dialog = SemanticEvaluateModelDialog(self)
        self.optimize_model_dialog = OptimizeModelDialog(self)
        self.classify_deploy_model_dialog = ClassifyDeployModelDialog(self)
        self.detect_deploy_model_dialog = DetectDeployModelDialog(self)
        self.segment_deploy_model_dialog = SegmentDeployModelDialog(self)
        self.semantic_deploy_model_dialog = SemanticDeployModelDialog(self)
        self.detect_video_inference_dialog = DetectVideoInferenceDialog(self)
        self.segment_video_inference_dialog = SegmentVideoInferenceDialog(self)

        # Create dialogs (SAM)
        self.sam_deploy_predictor_dialog = SAMDeployPredictorDialog(self)
        self.sam_deploy_generator_dialog = SAMDeployGeneratorDialog(self)

        # Create dialogs (See Anything)
        self.see_anything_train_model_dialog = SeeAnythingTrainModelDialog(self)
        self.see_anything_deploy_predictor_dialog = SeeAnythingDeployPredictorDialog(self)
        self.see_anything_deploy_generator_dialog = SeeAnythingDeployGeneratorDialog(self)

        # Create dialogs (Transformers)
        self.transformers_deploy_model_dialog = TransformersDeployModelDialog(self)

        # Create dialogs (Batch Inference - Consolidated)
        # This is accessed via ImageWindow right-click context menu
        self.batch_inference_dialog = BatchInferenceDialog(self)

        # Create dialogs (Tile)
        self.tile_manager_dialog = TileManagerDialog(self)
        self.classify_tile_dataset_dialog = ClassifyTileDatasetDialog(self)
        self.detect_tile_dataset_dialog = DetectTileDatasetDialog(self)
        self.segment_tile_dataset_dialog = SegmentTileDatasetDialog(self)
        self.semantic_tile_dataset_dialog = SemanticTileDatasetDialog(self)

        # Create dialogs (Break Time)
        self.snake_game_dialog = SnakeGame(self)
        self.breakout_game_dialog = BreakoutGame(self)
        self.lightcycle_game_dialog = LightCycleGame(self)

        # Connect signals to update status bar
        self.annotation_window.imageLoaded.connect(self.update_image_dimensions)
        self.annotation_window.mouseMoved.connect(self.update_mouse_position)
        self.annotation_window.viewChanged.connect(self.update_view_dimensions)

        # Connect the toolChanged signal (to the AnnotationWindow)
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        # Connect the toolChanged signal to the LabelWindow update_label_count_state method
        self.toolChanged.connect(self.label_window.update_annotation_count_state)
        # Connect the toolChanged signal (to the Toolbar)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)
        # Connect the selectedLabel signal to the LabelWindow's set_selected_label method
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        # Connect the annotationSelected to the LabelWindow's update_annotation_count
        self.annotation_window.annotationSelected.connect(self.label_window.update_annotation_count)
        # Connect the annotationCreated and annotationDeleted to update tooltips
        self.annotation_window.annotationCreated.connect(self.label_window.update_tooltips)
        self.annotation_window.annotationDeleted.connect(self.label_window.update_tooltips)
        # Connect the labelSelected signal from LabelWindow to update the selected label in AnnotationWindow
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)
        # Connect the labelSelected signal from LabelWindow to update the transparency slider
        self.label_window.transparencyChanged.connect(self.update_label_transparency)
        # Connect the imageSelected signal to update_current_image_path in AnnotationWindow
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        # Connect the imageChanged signal from ImageWindow to cancel SAM working area
        self.image_window.imageChanged.connect(self.handle_image_changed)
        # Connect the filterChanged signal from ImageWindow to expand ConfidenceWindow height
        self.image_window.filterGroupToggled.connect(self.on_image_window_filter_toggled)
        # Connect the zChannelRemoved signal from ImageWindow to update status bar
        self.image_window.zChannelRemoved.connect(self.on_z_channel_removed)
        # Connect the zChannelRemoved signal from ImageWindow to clear z-channel visualization in AnnotationWindow
        self.image_window.zChannelRemoved.connect(self.annotation_window.clear_z_channel_visualization)
        # Connect the imageLoaded signal from ImageWindow to check z-channel status
        self.image_window.imageLoaded.connect(self.on_image_loaded_check_z_channel)

        # ----------------------------------------
        # Create the menu bar
        # ----------------------------------------
        self.menu_bar = self.menuBar()

        # ========== FILE MENU ==========
        # File menu
        self.file_menu = self.menu_bar.addMenu("File")

        # Import menu
        self.import_menu = self.file_menu.addMenu("Import")

        # Raster submenu
        self.import_rasters_menu = self.import_menu.addMenu("Rasters")
        # Import Images
        self.import_images_action = QAction("Images", self)
        self.import_images_action.triggered.connect(self.import_images.import_images)
        self.import_rasters_menu.addAction(self.import_images_action)
        # Import Frames
        self.import_frames_action = QAction("Frames from Video", self)
        self.import_frames_action.triggered.connect(self.open_import_frames_dialog)
        self.import_rasters_menu.addAction(self.import_frames_action)

        # Labels submenu
        self.import_labels_menu = self.import_menu.addMenu("Labels")
        # Import Labels
        self.import_labels_action = QAction("Labels (JSON)", self)
        self.import_labels_action.triggered.connect(self.import_labels.import_labels)
        self.import_labels_menu.addAction(self.import_labels_action)
        # Import CoralNet Labels
        self.import_coralnet_labels_action = QAction("CoralNet Labels (CSV)", self)
        self.import_coralnet_labels_action.triggered.connect(self.import_coralnet_labels.import_coralnet_labels)
        self.import_labels_menu.addAction(self.import_coralnet_labels_action)
        # Import TagLab Labels
        self.import_taglab_labels_action = QAction("TagLab Labels (JSON)", self)
        self.import_taglab_labels_action.triggered.connect(self.import_taglab_labels.import_taglab_labels)
        self.import_labels_menu.addAction(self.import_taglab_labels_action)

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
        self.import_viscore_annotations_action.triggered.connect(self.open_import_viscore_annotations_dialog)
        self.import_annotations_menu.addAction(self.import_viscore_annotations_action)
        # Import TagLab Annotations
        self.import_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.import_taglab_annotations_action.triggered.connect(self.import_taglab_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_taglab_annotations_action)
        # Import Squidle Annotations
        self.import_squidle_annotations_action = QAction("Squidle (JSON)", self)
        self.import_squidle_annotations_action.triggered.connect(self.import_squidle_annotations.import_annotations)
        self.import_annotations_menu.addAction(self.import_squidle_annotations_action)

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
        self.export_menu = self.file_menu.addMenu("Export")

        # Labels submenu
        self.export_labels_menu = self.export_menu.addMenu("Labels")
        # Export Labels
        self.export_labels_action = QAction("Labels (JSON)", self)
        self.export_labels_action.triggered.connect(self.export_labels.export_labels)
        self.export_labels_menu.addAction(self.export_labels_action)
        # Export TagLab Labels
        self.export_taglab_labels_action = QAction("TagLab Labels (JSON)", self)
        self.export_taglab_labels_action.triggered.connect(self.export_taglab_labels.export_taglab_labels)
        self.export_labels_menu.addAction(self.export_taglab_labels_action)

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
        self.export_viscore_annotations_action = QAction("Viscore (CSV, JSON)", self)
        self.export_viscore_annotations_action.triggered.connect(self.open_export_viscore_annotations_dialog)
        self.export_annotations_menu.addAction(self.export_viscore_annotations_action)
        # Export TagLab Annotations
        self.export_taglab_annotations_action = QAction("TagLab (JSON)", self)
        self.export_taglab_annotations_action.triggered.connect(self.export_taglab_annotations.export_annotations)
        self.export_annotations_menu.addAction(self.export_taglab_annotations_action)
        # Export GeoJSON Annotations
        self.export_geojson_annotations_action = QAction("GeoJSON (JSON)", self)
        self.export_geojson_annotations_action.triggered.connect(self.export_geojson_annotations_dialog.exec_)
        self.export_annotations_menu.addAction(self.export_geojson_annotations_action)
        # Export Mask Annotations
        self.export_mask_annotations_action = QAction("Masks (Raster)", self)
        self.export_mask_annotations_action.triggered.connect(self.open_export_mask_annotations_dialog)
        self.export_annotations_menu.addAction(self.export_mask_annotations_action)

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
        # Export Semantic Segmentation Dataset
        self.export_semantic_dataset_action = QAction("Semantic", self)
        self.export_semantic_dataset_action.triggered.connect(self.open_semantic_export_dataset_dialog)
        self.export_dataset_menu.addAction(self.export_semantic_dataset_action)

        # Add a separator
        self.file_menu.addSeparator()

        # New Project
        self.new_project_action = QAction("New Project", self)
        self.new_project_action.triggered.connect(self.open_new_project)
        self.file_menu.addAction(self.new_project_action)
        # Open Project
        self.open_project_action = QAction("Open Project (JSON)", self)
        self.open_project_action.triggered.connect(self.open_open_project_dialog)
        self.file_menu.addAction(self.open_project_action)
        # Save Project
        self.save_project_action = QAction("Save Project (JSON)", self)
        self.save_project_action.setToolTip("Ctrl + Shift + S")
        self.save_project_action.triggered.connect(self.open_save_project_dialog)
        self.file_menu.addAction(self.save_project_action)

        # ========== UTILITIES MENU ==========
        # Utilities menu
        self.utilities_menu = self.menu_bar.addMenu("Utilities")
        
        # Sampling Annotations
        self.annotation_sampling_action = QAction("Sample", self)
        self.annotation_sampling_action.triggered.connect(self.open_patch_annotation_sampling_dialog)
        self.utilities_menu.addAction(self.annotation_sampling_action)
        
        # Add a separator
        self.utilities_menu.addSeparator()
        
        # Tile submenu
        self.tile_menu = self.utilities_menu.addMenu("Tile")
        
        # Tile Creation
        self.tile_manager_action = QAction("Tile Manager", self)
        self.tile_manager_action.triggered.connect(self.open_tile_manager_dialog)
        self.tile_menu.addAction(self.tile_manager_action)
        
        # Add a separator
        self.tile_menu.addSeparator()
        
        # Tile Dataset submenu
        self.tile_dataset_menu = self.tile_menu.addMenu("Tile Dataset")
        # Tile Classify Dataset
        self.classify_tile_dataset_action = QAction("Classify", self)
        self.classify_tile_dataset_action.triggered.connect(self.open_classify_tile_dataset_dialog)
        self.tile_dataset_menu.addAction(self.classify_tile_dataset_action)
        # Tile Detect Dataset
        self.detect_tile_dataset_action = QAction("Detect", self)
        self.detect_tile_dataset_action.triggered.connect(self.open_detect_tile_dataset_dialog)
        self.tile_dataset_menu.addAction(self.detect_tile_dataset_action)
        # Tile Segment Dataset
        self.segment_tile_dataset_action = QAction("Segment", self)
        self.segment_tile_dataset_action.triggered.connect(self.open_segment_tile_dataset_dialog)
        self.tile_dataset_menu.addAction(self.segment_tile_dataset_action)
        # Tile Semantic Dataset
        self.semantic_tile_dataset_action = QAction("Semantic", self)
        self.semantic_tile_dataset_action.triggered.connect(self.open_semantic_tile_dataset_dialog)
        self.tile_dataset_menu.addAction(self.semantic_tile_dataset_action)

        # ========== AI-ASSIST MENU ==========
        # AI-Assist menu
        self.ai_assist_menu = self.menu_bar.addMenu("AI-Assist")
        
        # SAM submenu
        self.sam_menu = self.ai_assist_menu.addMenu("SAM")
        # Deploy Predictor
        self.sam_deploy_model_action = QAction("Deploy Predictor", self)
        self.sam_deploy_model_action.triggered.connect(self.open_sam_deploy_predictor_dialog)
        self.sam_menu.addAction(self.sam_deploy_model_action)
        # Deploy Generator
        self.sam_deploy_generator_action = QAction("Deploy Generator", self)
        self.sam_deploy_generator_action.triggered.connect(self.open_sam_deploy_generator_dialog)
        self.sam_menu.addAction(self.sam_deploy_generator_action)

        # See Anything submenu
        self.see_anything_menu = self.ai_assist_menu.addMenu("See Anything")
        # Train Model
        self.see_anything_train_model_action = QAction("Train Model", self)
        self.see_anything_train_model_action.triggered.connect(self.open_see_anything_train_model_dialog)
        self.see_anything_menu.addAction(self.see_anything_train_model_action)
        # Add separator
        self.see_anything_menu.addSeparator()
        # Deploy Predictor
        self.see_anything_deploy_predictor_action = QAction("Deploy Predictor", self)
        self.see_anything_deploy_predictor_action.triggered.connect(self.open_see_anything_deploy_predictor_dialog)
        self.see_anything_menu.addAction(self.see_anything_deploy_predictor_action)
        # Deploy Generator
        self.see_anything_deploy_generator_action = QAction("Deploy Generator", self)
        self.see_anything_deploy_generator_action.triggered.connect(self.open_see_anything_deploy_generator_dialog)
        self.see_anything_menu.addAction(self.see_anything_deploy_generator_action)

        # Transformers submenu
        self.transformers_menu = self.ai_assist_menu.addMenu("Transformers")
        # Deploy Model
        self.transformers_deploy_model_action = QAction("Deploy Model", self)
        self.transformers_deploy_model_action.triggered.connect(self.open_transformers_deploy_model_dialog)
        self.transformers_menu.addAction(self.transformers_deploy_model_action)

        # ========== MACHINE LEARNING MENU ==========
        # Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

        # Merge Datasets submenu
        self.ml_merge_datasets_menu = self.ml_menu.addMenu("Merge Datasets")
        # Merge Classification Datasets
        self.ml_classify_merge_datasets_action = QAction("Classify", self)
        self.ml_classify_merge_datasets_action.triggered.connect(self.open_classify_merge_datasets_dialog)
        self.ml_merge_datasets_menu.addAction(self.ml_classify_merge_datasets_action)
        
        # Tune Model submenu
        self.ml_tune_model_menu = self.ml_menu.addMenu("Tune Model")
        # Tune Classification Model
        self.ml_classify_tune_model_action = QAction("Classify", self)
        self.ml_classify_tune_model_action.triggered.connect(self.open_classify_tune_model_dialog)
        self.ml_tune_model_menu.addAction(self.ml_classify_tune_model_action)
        # Tune Detection Model
        self.ml_detect_tune_model_action = QAction("Detect", self)
        self.ml_detect_tune_model_action.triggered.connect(self.open_detect_tune_model_dialog)
        self.ml_tune_model_menu.addAction(self.ml_detect_tune_model_action)
        # Tune Segmentation Model
        self.ml_segment_tune_model_action = QAction("Segment", self)
        self.ml_segment_tune_model_action.triggered.connect(self.open_segment_tune_model_dialog)
        self.ml_tune_model_menu.addAction(self.ml_segment_tune_model_action)
        # Tune Semantic Segmentation Model
        self.ml_semantic_tune_model_action = QAction("Semantic", self)
        self.ml_semantic_tune_model_action.triggered.connect(self.open_semantic_tune_model_dialog)
        # self.ml_tune_model_menu.addAction(self.ml_semantic_tune_model_action)
        
        # Add a separator
        self.ml_menu.addSeparator()

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
        # Train Semantic Segmentation Model
        self.ml_semantic_train_model_action = QAction("Semantic", self)
        self.ml_semantic_train_model_action.triggered.connect(self.open_semantic_train_model_dialog)
        self.ml_train_model_menu.addAction(self.ml_semantic_train_model_action)

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
        # Evaluate Semantic Segmentation Model
        self.ml_semantic_evaluate_model_action = QAction("Semantic", self)
        self.ml_semantic_evaluate_model_action.triggered.connect(self.open_semantic_evaluate_model_dialog)
        self.ml_evaluate_model_menu.addAction(self.ml_semantic_evaluate_model_action)

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
        # Deploy Semantic Segmentation Model
        self.ml_semantic_deploy_model_action = QAction("Semantic", self)
        self.ml_semantic_deploy_model_action.triggered.connect(self.open_semantic_deploy_model_dialog)
        self.ml_deploy_model_menu.addAction(self.ml_semantic_deploy_model_action)
        
        # Add a separator
        self.ml_menu.addSeparator()

        # Video Inference submenu
        self.ml_video_inference_menu = self.ml_menu.addMenu("Video Inference")
        # Video Inference Detection
        self.ml_detect_video_inference_action = QAction("Detect", self)
        self.ml_detect_video_inference_action.triggered.connect(self.open_detect_video_inference_dialog)
        self.ml_video_inference_menu.addAction(self.ml_detect_video_inference_action)
        # Video Inference Segmentation
        self.ml_segment_video_inference_action = QAction("Segment", self)
        self.ml_segment_video_inference_action.triggered.connect(self.open_segment_video_inference_dialog)
        self.ml_video_inference_menu.addAction(self.ml_segment_video_inference_action)

        # ========== CORALNET MENU ==========
        # CoralNet menu
        self.coralnet_menu = self.menu_bar.addMenu("CoralNet")
        
        # CoralNet Authenticate
        self.coralnet_authenticate_action = QAction("Authenticate", self)
        self.coralnet_authenticate_action.triggered.connect(self.open_coralnet_authenticate_dialog)
        self.coralnet_menu.addAction(self.coralnet_authenticate_action)
        # CoralNet Download
        self.coralnet_download_action = QAction("Download", self)
        self.coralnet_download_action.triggered.connect(self.open_coralnet_download_dialog)
        self.coralnet_menu.addAction(self.coralnet_download_action)

        # ========== EXPLORER ACTION ==========
        # Explorer action
        self.open_explorer_action = QAction("Explorer", self)
        self.open_explorer_action.triggered.connect(self.open_explorer_window)
        self.menu_bar.addAction(self.open_explorer_action)

        # ========== HELP MENU ==========
        # Help menu
        self.help_menu = self.menu_bar.addMenu("Help")

        # Check for updates
        self.check_for_updates_action = QAction("Check for Updates", self)
        self.check_for_updates_action.triggered.connect(self.open_check_for_updates_dialog)
        self.help_menu.addAction(self.check_for_updates_action)
        # Usage
        self.usage_action = QAction("Usage / Hotkeys", self)
        self.usage_action.triggered.connect(self.open_usage_dialog)
        self.help_menu.addAction(self.usage_action)
        # System Monitor
        self.system_monitor_action = QAction("System Monitor", self)
        self.system_monitor_action.triggered.connect(self.open_system_monitor_dialog)
        self.help_menu.addAction(self.system_monitor_action)
        # Issues / Feature Requests
        self.create_issue_action = QAction("Issues / Feature Requests", self)
        self.create_issue_action.triggered.connect(self.open_create_new_issue_dialog)
        self.help_menu.addAction(self.create_issue_action)
        # Separator
        self.help_menu.addSeparator()
        # Create Break Time submenu
        break_time_menu = self.help_menu.addMenu("Take a Break")
        # Snake Game
        snake_game_action = QAction("Snake Game", self)
        snake_game_action.triggered.connect(self.open_snake_game_dialog)
        break_time_menu.addAction(snake_game_action)
        # Break Out Game
        break_out_game_action = QAction("Breakout Game", self)
        break_out_game_action.triggered.connect(self.open_breakout_game_dialog)
        break_time_menu.addAction(break_out_game_action)
        # Light Cycle Game
        light_cycle_game_action = QAction("Light Cycle Game", self)
        light_cycle_game_action.triggered.connect(self.open_light_cycle_game_dialog)
        break_time_menu.addAction(light_cycle_game_action)

        # ----------------------------------------
        # Create and add the toolbar
        # ----------------------------------------
        
        # Define verbose tool descriptions
        self.tool_descriptions = {
            "select": ("Select Tool\n\n"
                       "Select, modify, and manage annotations.\n"
                       "• Left-click to select annotations; hold Ctrl+left-click to select multiple.\n"
                       "• Left-click and drag to move selected annotations.\n"
                       "• Ctrl+click and drag to create a selection rectangle.\n"
                       "• Ctrl+Shift to show resize handles for a selected Rectangle and Polygon annotations.\n"
                       "• Ctrl+X to cut a selected annotation along a drawn line.\n"
                       "• Ctrl+C to combine multiple selected annotations.\n"
                       "• Ctrl+Space to confirm selected annotations with top predictions.\n"
                       "• Ctrl+Shift+mouse wheel to adjust polygon complexity.\n"
                       "• Ctrl+Delete to remove selected annotations."),
            
            "scale": ("Scale Tool\n\n"
                      "Provide scale to the image(s), and measure distances on the current image.\n"
                      "• Left-click to set the starting point.\n"
                      "• Drag to draw a line, then left-click again to set the endpoint.\n"
                      "• Press Backspace to cancel drawing the scale line."
                      "• The scale will be calculated based on the known provided length and pixel length.\n"
                      "• Area and Perimeter for an annotation can be viewed when hovering over the Confidence Window.\n"
                      "• Preferred units can be set in the Status Bar."),

            "patch": ("Patch Tool\n\n"
                      "Create point (patch) annotations centered at the cursor.\n"
                      "• Left-click to place a patch at the mouse location.\n"
                      "• Hold Ctrl and use the mouse wheel or use the Patch Size box to adjust patch size.\n"
                      "• A semi-transparent preview shows the patch before placing it."),
            
            "rectangle": ("Rectangle Tool\n\n"
                          "Create rectangular annotations by clicking and dragging.\n"
                          "• Left-click to set the first corner, then move the mouse to size the rectangle.\n"
                          "• Left-click again to place the rectangle.\n"
                          "• Press Backspace to cancel drawing the current rectangle.\n"
                          "• A semi-transparent preview shows the rectangle while drawing."),
            
            "polygon": ("Polygon Tool\n\n"
                        "Create polygon annotations with multiple vertices.\n"
                        "• Left-click to set the first vertex, then move the mouse to draw the polygon\n"
                        "• Hold Ctrl while left-clicking to draw straight-line segments.\n"
                        "• Left-click again to complete the polygon.\n"
                        "• Press Backspace to cancel the current polygon.\n"
                        "• A semi-transparent preview shows the polygon while drawing."),
            
            "brush": ("Brush Tool\n\n"
                      "Create freehand brush annotations by clicking and dragging.\n"
                      "• Left-click and drag to paint brush strokes on the canvas.\n"
                      "• Hold Ctrl and use the mouse wheel to adjust brush size.\n"
                      "• Press Ctrl + Shift to switch between a circle and square brush shape.\n"
                      "• A semi-transparent preview shows the brush stroke while drawing."),

            "erase": ("Erase Tool\n\n"
                      "Erase pixels from mask annotations.\n"
                      "• Left-click and drag to erase pixels.\n"
                      "• Hold Ctrl and use the mouse wheel to adjust eraser size.\n"
                      "• Press Ctrl + Shift to switch between a circle and square eraser shape.\n"
                      "• Press Ctrl + (Backspace or Delete) to clear the mask annotation on the current image.\n"
                      "• A semi-transparent preview shows the eraser while drawing."),

            "fill": ("Fill Tool\n\n"
                     "Fill contiguous regions in mask annotations.\n"
                     "• Left-click to fill the region under the cursor with the selected label."),

            "dropper": ("Dropper Tool\n\n"
                        "Select a label from the current mask annotation.\n"
                        "• Left-click on a pixel in the mask to select its label in the label window."),

            "sam": ("Segment Anything (SAM) Tool\n\n"
                    "Generates AI-powered segmentations.\n"
                    "• Left-click to create a working area, then left-click again to confirm.\n"
                    "\t• Or, press Spacebar to create a working area for the current view.\n"
                    "• Ctrl+Left-click to add positive points (foreground).\n"
                    "• Ctrl+Right-click to add negative points (background).\n"
                    "• Left-click and drag to create a bounding box for prompting.\n"
                    "• Press Spacebar to generate and confirm the segmentation.\n"
                    "• Press Backspace to cancel the current operation.\n"
                    "• Uncertainty can be adjusted in Parameters section.\n"
                    "• A SAM predictor must be deployed first."),
            
            "see_anything": ("See Anything (YOLOE) Tool\n\n"
                             "Uses YOLOE to detect / segments objects of interest based on visual prompts.\n"
                             "• Left-click to create a working area, then click again to confirm.\n"
                             "\t• Or, press Spacebar to create a working area for the current view.\n"
                             "• Draw rectangles inside the working area to guide detection.\n"
                             "• Press Spacebar to generate detections using drawn rectangles.\n"
                             "• Press Spacebar again to confirm annotations or apply SAM refinement.\n"
                             "• Press Backspace to cancel current operation or clear annotations.\n"
                             "• Uncertainty can be adjusted in Parameters section.\n"
                             "• A See Anything (YOLOE) predictor must be deployed first."),
            
            "work_area": ("Work Area Tool\n\n"
                          "Defines regions for detection and segmentation models to run predictions on.\n"
                          "• Left-click to create a working area, then left-click again to confirm.\n"
                          "\t• Or, press Spacebar to create a work area from the current view.\n"
                          "• Hold Ctrl+Shift to show delete buttons for existing work areas.\n"
                          "• Press Ctrl+Shift+Backspace to clear all work areas.\n"
                          "• Hold Ctrl+Alt to temporarily view a work area of the current view.\n"
                          "• Work areas can be used with Tile Batch Inference and other batch operations.\n"
                          "• All work areas are automatically saved with the image in a Project (JSON) file."),
            
        }
    
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
        self.select_tool_action.setToolTip(self.tool_descriptions["select"])
        self.select_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.select_tool_action)

        self.toolbar.addSeparator()

        self.scale_tool_action = QAction(self.scale_icon, "Scale", self)
        self.scale_tool_action.setCheckable(True)
        self.scale_tool_action.setToolTip(self.tool_descriptions["scale"])
        self.scale_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.scale_tool_action)

        self.toolbar.addSeparator()
        
        self.patch_tool_action = QAction(self.patch_icon, "Patch", self)
        self.patch_tool_action.setCheckable(True)
        self.patch_tool_action.setToolTip(self.tool_descriptions["patch"])
        self.patch_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.patch_tool_action)

        self.rectangle_tool_action = QAction(self.rectangle_icon, "Rectangle", self)
        self.rectangle_tool_action.setCheckable(True)
        self.rectangle_tool_action.setToolTip(self.tool_descriptions["rectangle"])
        self.rectangle_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.rectangle_tool_action)

        self.polygon_tool_action = QAction(self.polygon_icon, "Polygon", self)
        self.polygon_tool_action.setCheckable(True)
        self.polygon_tool_action.setToolTip(self.tool_descriptions["polygon"])
        self.polygon_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.polygon_tool_action)

        self.toolbar.addSeparator()

        self.brush_tool_action = QAction(self.brush_icon, "Brush", self)
        self.brush_tool_action.setCheckable(True)
        self.brush_tool_action.setToolTip(self.tool_descriptions["brush"])
        self.brush_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.brush_tool_action)

        self.erase_tool_action = QAction(self.erase_icon, "Erase", self)
        self.erase_tool_action.setCheckable(True)
        self.erase_tool_action.setToolTip(self.tool_descriptions["erase"])
        self.erase_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.erase_tool_action)

        self.dropper_tool_action = QAction(self.dropper_icon, "Dropper", self)
        self.dropper_tool_action.setCheckable(True)
        self.dropper_tool_action.setToolTip(self.tool_descriptions["dropper"])
        self.dropper_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.dropper_tool_action)

        self.fill_tool_action = QAction(self.fill_icon, "Fill", self)
        self.fill_tool_action.setCheckable(True)
        self.fill_tool_action.setToolTip(self.tool_descriptions["fill"])
        self.fill_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.fill_tool_action)

        self.toolbar.addSeparator()

        self.sam_tool_action = QAction(self.sam_icon, "SAM", self)
        self.sam_tool_action.setCheckable(True)
        self.sam_tool_action.setToolTip(self.tool_descriptions["sam"])
        self.sam_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.sam_tool_action)

        self.see_anything_tool_action = QAction(self.see_anything_icon, "See Anything (YOLOE)", self)
        self.see_anything_tool_action.setCheckable(True)
        self.see_anything_tool_action.setToolTip(self.tool_descriptions["see_anything"])
        self.see_anything_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.see_anything_tool_action)

        self.toolbar.addSeparator()

        self.work_area_tool_action = QAction(self.workarea_icon, "Work Area", self)
        self.work_area_tool_action.setCheckable(True)
        self.work_area_tool_action.setToolTip(self.tool_descriptions["work_area"])
        self.work_area_tool_action.triggered.connect(self.toggle_tool)
        self.toolbar.addAction(self.work_area_tool_action)
        
        self.toolbar.addSeparator()

        # Add a spacer to push the device label to the bottom
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)

        # Add the device label widget as an action in the toolbar
        self.devices = self.get_available_devices()
        # Get the 'best' device available
        self.device = self.devices[-1]

        if self.device.startswith('cuda'):
            if len([d for d in self.devices if d.endswith('cuda')]) > 1:
                device_icon = self.rocket_icon
            else:
                device_icon = self.rabbit_icon
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

        # Labels for project, image dimensions and mouse position
        self.mouse_position_label = QLabel("Mouse: X: 0, Y: 0")
        self.mouse_position_label.setFixedWidth(150)
        
        self.image_dimensions_label = QLabel("Image: 0 x 0")
        self.image_dimensions_label.setFixedWidth(150)

        self.view_dimensions_label = QLabel("View: 0 x 0")
        self.view_dimensions_label.setFixedWidth(150)
        
        self.scaled_view_prefix_label = QLabel("Scale:")
        self.scaled_view_prefix_label.setEnabled(False)  # Disabled by default

        self.scaled_view_dims_label = QLabel("0 x 0")
        self.scaled_view_dims_label.setFixedWidth(120)  # For "height x width"
        self.scaled_view_dims_label.setEnabled(False)   # Disabled by default
        
        self.scale_unit_dropdown = QComboBox()
        self.scale_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'])
        self.scale_unit_dropdown.setCurrentIndex(2)  # Default to 'm'
        self.scale_unit_dropdown.setFixedWidth(60)
        self.scale_unit_dropdown.setEnabled(False)  # Disabled by default

        # Slider
        transparency_layout = QHBoxLayout()
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 255)
        self.transparency_slider.setValue(128)  # Default transparency
        self.transparency_slider.setTickPosition(QSlider.TicksBelow)
        self.transparency_slider.setTickInterval(16)  # Add tick marks every 16 units
        self.transparency_slider.valueChanged.connect(self.update_label_transparency)

        # Left icon (transparent)
        transparent_icon = QLabel()
        transparent_icon.setPixmap(self.transparent_icon.pixmap(QSize(16, 16)))
        transparent_icon.setToolTip("Transparent")

        # Right icon (opaque)
        opaque_icon = QLabel()
        opaque_icon.setPixmap(self.opaque_icon.pixmap(QSize(16, 16)))
        opaque_icon.setToolTip("Opaque")

        # Add widgets to the transparency layout
        transparency_layout.addWidget(transparent_icon)
        transparency_layout.addWidget(self.transparency_slider)
        transparency_layout.addWidget(opaque_icon)

        # Create widget to hold the layout
        self.transparency_widget = QWidget()
        self.transparency_widget.setLayout(transparency_layout)

        # Z unit dropdown
        self.z_unit_dropdown = QComboBox()
        self.z_unit_dropdown.addItems(['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi', 'px'])
        self.z_unit_dropdown.setCurrentIndex(2)  # Default to 'm'
        self.z_unit_dropdown.setFixedWidth(60)
        self.z_unit_dropdown.setEnabled(False)  # Disabled by default until Z data is available
        
        # Z label for depth information
        self.z_label = QLabel("Z: -----")
        self.z_label.setEnabled(False)  # Disabled by default until Z data is available

        # Z colormap dropdown for visualization
        self.z_colormap_dropdown = QComboBox()
        self.z_colormap_dropdown.addItems([
            'None', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo'
        ])
        self.z_colormap_dropdown.setCurrentText('None')
        self.z_colormap_dropdown.setFixedWidth(100)
        self.z_colormap_dropdown.setEnabled(False)  # Disabled by default until Z data is available
        self.z_colormap_dropdown.setToolTip("Select colormap for Z-channel visualization")

        # Z dynamic scaling button
        self.z_dynamic_button = QToolButton()
        self.z_dynamic_button.setCheckable(True)
        self.z_dynamic_button.setChecked(False)
        self.z_dynamic_button.setIcon(self.dynamic_icon)
        self.z_dynamic_button.setToolTip("Toggle dynamic Z-range scaling based on visible area")
        self.z_dynamic_button.setEnabled(False)  # Disabled by default until Z data is available
        
        # Z button and Z label
        self.z_action = QAction(self.z_icon, "", self)
        self.z_action.setCheckable(False)  # TODO
        self.z_action.setChecked(False)
        self.z_action.setToolTip("Depth Estimation (In Progress)")
        # self.z_action.triggered.connect(self.open_depth_dialog)  # TODO Disabled for now

        # Create button to hold the Z action
        self.z_button = QToolButton()
        self.z_button.setDefaultAction(self.z_action)

        # Patch Annotation Size
        annotation_size_label = QLabel("Patch Size")
        self.annotation_size_spinbox = QSpinBox()
        self.annotation_size_spinbox.setMinimum(1)
        self.annotation_size_spinbox.setMaximum(5000)
        self.annotation_size_spinbox.setEnabled(False)
        self.annotation_size_spinbox.setValue(self.annotation_window.annotation_size)
        self.annotation_size_spinbox.valueChanged.connect(self.annotation_window.set_annotation_size)
        self.annotation_window.annotationSizeChanged.connect(self.annotation_size_spinbox.setValue)

        annotation_size_layout = QHBoxLayout()
        annotation_size_layout.addWidget(annotation_size_label)
        annotation_size_layout.addWidget(self.annotation_size_spinbox)

        self.annotation_size_widget = QWidget()
        self.annotation_size_widget.setLayout(annotation_size_layout)

        # --------------------------------------------------
        # Create collapsible Parameters section
        # --------------------------------------------------
        self.parameters_section = CollapsibleSection("Parameters")

        # Max detections spinbox
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setRange(1, 10000)
        self.max_detections_spinbox.setValue(self.max_detections)
        self.max_detections_spinbox.valueChanged.connect(self.update_max_detections)
        max_detections_layout = QHBoxLayout()
        max_detections_label = QLabel("Max Detections:")
        max_detections_layout.addWidget(max_detections_label)
        max_detections_layout.addWidget(self.max_detections_spinbox)
        max_detections_layout.addStretch()
        max_detections_widget = QWidget()
        max_detections_widget.setLayout(max_detections_layout)
        self.parameters_section.add_widget(max_detections_widget, "Max Detections")

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
        self.area_threshold_min_slider = QSlider(Qt.Horizontal)
        self.area_threshold_min_slider.setMinimum(0)
        self.area_threshold_min_slider.setMaximum(100)
        self.area_threshold_min_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_min_slider.setTickInterval(10)
        self.area_threshold_min_slider.setValue(int(min_val * 100))
        self.area_threshold_min_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_max_slider = QSlider(Qt.Horizontal)
        self.area_threshold_max_slider.setMinimum(0)
        self.area_threshold_max_slider.setMaximum(100)
        self.area_threshold_max_slider.setTickPosition(QSlider.TicksBelow)
        self.area_threshold_max_slider.setTickInterval(10)
        self.area_threshold_max_slider.setValue(int(max_val * 100))
        self.area_threshold_max_slider.valueChanged.connect(self.update_area_label)
        self.area_threshold_label = QLabel(f"{min_val:.2f} - {max_val:.2f}")
        area_thresh_layout = QVBoxLayout()
        area_thresh_layout.addWidget(self.area_threshold_min_slider)
        area_thresh_layout.addWidget(self.area_threshold_max_slider)
        area_thresh_layout.addWidget(self.area_threshold_label)
        area_thresh_widget = QWidget()
        area_thresh_widget.setLayout(area_thresh_layout)
        self.parameters_section.add_widget(area_thresh_widget, "Area Threshold")

        # Add widgets to status bar layout
        self.status_bar_layout.addWidget(self.mouse_position_label)
        self.status_bar_layout.addWidget(self.image_dimensions_label)
        self.status_bar_layout.addWidget(self.view_dimensions_label)
        self.status_bar_layout.addWidget(self.transparency_widget)
        self.status_bar_layout.addWidget(self.scale_unit_dropdown)
        self.status_bar_layout.addWidget(self.scaled_view_prefix_label)
        self.status_bar_layout.addWidget(self.scaled_view_dims_label)
        self.status_bar_layout.addWidget(self.z_unit_dropdown)
        self.status_bar_layout.addWidget(self.z_label)
        self.status_bar_layout.addWidget(self.z_colormap_dropdown)
        self.status_bar_layout.addWidget(self.z_dynamic_button)
        self.status_bar_layout.addWidget(self.z_button)
        self.status_bar_layout.addWidget(self.annotation_size_widget)
        self.status_bar_layout.addWidget(self.parameters_section)

        # --------------------------------------------------
        # Create the main layout
        # --------------------------------------------------
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main vertical layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Status bar in a group box
        self.status_bar_group_box = QGroupBox("Status Bar")
        self.status_bar_group_box.setLayout(self.status_bar_layout)
        self.main_layout.addWidget(self.status_bar_group_box)

        # Panels layout: horizontal row under status bar 
        # (LabelWindow, AnnotationWindow, ImageWindow + ConfidenceWindow)
        self.panels_layout = QHBoxLayout()

        # Label panel (left)
        self.label_layout = QVBoxLayout()
        self.label_layout.addWidget(self.label_window)

        # Add the timer group box under the label window (which contains Counts)
        self.timer_group = TimerGroupBox(self)
        self.label_layout.addWidget(self.timer_group)

        # Annotation panel (center) (in a group box since it's a QGraphicsView)
        self.annotation_layout = QVBoxLayout()
        self.annotation_group_box = QGroupBox("Annotation Window")
        group_layout = QVBoxLayout(self.annotation_group_box)
        group_layout.addWidget(self.annotation_window)
        self.annotation_group_box.setLayout(group_layout)
        self.annotation_layout.addWidget(self.annotation_group_box)

        # Image panel (ImageWindow + ConfidenceWindow stacked vertically)
        self.image_layout = QVBoxLayout()
        self.image_layout.addWidget(self.image_window, 54)
        self.image_layout.addWidget(self.confidence_window, 46)
        
        # Set stretch factors to control relative sizes
        self.panels_layout.addLayout(self.label_layout, 15)  # Strict width
        self.panels_layout.addLayout(self.annotation_layout, 120)  # Strict width
        self.panels_layout.addLayout(self.image_layout, 25)

        # Add the panels row to the main layout
        self.main_layout.addLayout(self.panels_layout)

        # --------------------------------------------------
        # Setup global event filter for shortcuts
        # --------------------------------------------------
        self.global_event_filter = GlobalEventFilter(self)
        QApplication.instance().installEventFilter(self.global_event_filter)

        # --------------------------------------------------
        # Enable drag and drop
        # --------------------------------------------------
        self.setAcceptDrops(True)
        
        # --------------------------------------------------
        # Update the scaled view dimensions label
        # --------------------------------------------------
        self.scale_unit_dropdown.currentTextChanged.connect(self.on_scale_unit_changed)
        self.z_unit_dropdown.currentTextChanged.connect(self.on_z_unit_changed)
        self.z_colormap_dropdown.currentTextChanged.connect(self.on_z_colormap_changed)
        self.z_dynamic_button.toggled.connect(self.on_z_dynamic_toggled)

        # --------------------------------------------------
        # Check for updates on opening
        # --------------------------------------------------
        self.open_check_for_updates_dialog(on_open=True)
        
        # Process events
        QApplication.processEvents()

    def showEvent(self, event):
        """Show the main window maximized."""
        super().showEvent(event)
        self.showMaximized()

    def closeEvent(self, event):
        """Ensure the explorer window and system monitor are closed when the main window closes."""
        if self.explorer_window:
            # Setting parent to None prevents it from being deleted with main window
            # before it can be properly handled.
            self.explorer_window.setParent(None)
            self.explorer_window.close()
        
        # Close the system monitor if it exists
        if self.system_monitor:
            self.system_monitor.close()
        
        # Stop timer threads properly
        if hasattr(self, 'timer_group') and self.timer_group:
            if hasattr(self.timer_group, 'timer_widget') and self.timer_group.timer_widget:
                self.timer_group.timer_widget.stop_threads()
            
        super().closeEvent(event)

    def changeEvent(self, event):
        """Handle window state changes (minimize, maximize, restore)."""
        super().changeEvent(event)
        if (event.type() == QEvent.WindowStateChange):
            if self.windowState() & Qt.WindowMinimized:
                # Allow minimizing
                pass
            elif self.windowState() & Qt.WindowMaximized:
                # Window is maximized, do nothing
                pass
            else:
                # Restore to normal state
                pass  # Do nothing, let the OS handle the restore

    def dragEnterEvent(self, event):
        """Handle drag enter event for drag-and-drop."""
        self.untoggle_all_tools()

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]

            # Accept if any of the files is a JSON file
            if any(file_name.lower().endswith('.json') for file_name in file_names):
                event.acceptProposedAction()
            else:
                self.import_images.dragEnterEvent(event)

    def dropEvent(self, event):
        """Handle drop event for drag-and-drop."""
        self.untoggle_all_tools()

        urls = event.mimeData().urls()
        file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]

        if file_names:
            # Check if a single JSON file was dropped
            if len(file_names) == 1 and file_names[0].lower().endswith('.json'):
                # Open as a project file
                path = file_names[0]
                self.open_project_dialog.file_path_edit.setText(path)
                self.open_project_dialog.load_selected_project()
                self.current_project_path = self.open_project_dialog.current_project_path
                self.update_project_label()
            else:
                # Handle as image imports
                self.import_images.dropEvent(event)

    def dragMoveEvent(self, event):
        """Handle drag move event for drag-and-drop."""
        self.untoggle_all_tools()

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]

            # Accept if any of the files is a JSON file
            if any(file_name.lower().endswith('.json') for file_name in file_names):
                event.acceptProposedAction()
            else:
                self.import_images.dragMoveEvent(event)
                
    def set_main_window_enabled_state(self, enable_list=None, disable_list=None):
        """
        Modular method to enable/disable widgets and actions in the main window.
        - enable_list: list of widgets/actions to enable
        - disable_list: list of widgets/actions to disable
        If both are None, enables everything.
        """
        # All main widgets/actions to consider
        all_widgets = [
            self.toolbar,
            self.menu_bar,
            self.image_window,
            self.label_window,
            self.confidence_window,
            self.annotation_window,
            # Status bar widgets
            *(self.status_bar_layout.itemAt(i).widget() for i in range(self.status_bar_layout.count()))
        ]
        # Remove None entries (in case any status bar slot is empty)
        all_widgets = [w for w in all_widgets if w is not None]

        # If neither list is provided, enable everything
        if enable_list is None and disable_list is None:
            for w in all_widgets:
                w.setEnabled(True)
            return

        # Disable everything by default
        for w in all_widgets:
            w.setEnabled(False)
            
        # Enable specified widgets/actions
        if enable_list:
            for w in enable_list:
                if w is not None:
                    w.setEnabled(True)
                    
        # Disable specified widgets/actions (overrides enable if both present)
        if disable_list:
            for w in disable_list:
                if w is not None:
                    w.setEnabled(False)
                
    def switch_back_to_tool(self):
        """Switches back to the tool used to create the currently selected annotation."""        
        # Get the currently selected tool from AnnotationWindow
        selected_tool = self.annotation_window.get_selected_tool()
        
        # Semantic-specific annotation tools toggle
        if selected_tool == 'brush':
            self.choose_specific_tool('erase')
            return
        elif selected_tool == 'erase':
            self.choose_specific_tool('brush')
            return
        elif selected_tool == 'fill':
            self.choose_specific_tool('brush')
            return
        elif selected_tool == 'dropper':
            self.choose_specific_tool('brush')
            return
                
        if selected_tool != "select":
            self.choose_specific_tool("select")
            return
        
        # Get the currently selected annotation type
        annotation_type = self.annotation_window.get_selected_annotation_type()
        
        if annotation_type is None:
            return
        
        # Convert the annotation type to a string
        annotation_type = str(annotation_type.__name__)
        
        if annotation_type == "PatchAnnotation":
            self.choose_specific_tool("patch")
        elif annotation_type == "RectangleAnnotation":
            self.choose_specific_tool("rectangle")
        elif annotation_type == "PolygonAnnotation":
            self.choose_specific_tool("polygon")
        elif annotation_type == "MultiPolygonAnnotation":
            self.choose_specific_tool("polygon")
        else:
            # Multiple annotations selected
            pass
        
    def choose_specific_tool(self, tool):
        """Choose a specific tool based on the provided tool name."""
        # Untoggle all tools first (clear buttons)
        self.untoggle_all_tools()
        # Trigger the select tool action in the main window to set the button
        self.select_tool_action.trigger()
        # Switch to select tool in main window (sets button)
        self.handle_tool_changed(tool)
        # Set the select tool in the annotation window (sets tool)
        self.annotation_window.set_selected_tool(tool)
        
    def toggle_tool(self, state):
        """Toggle the selected tool and emit the toolChanged signal."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "Image Not Loaded",
                                "Please load an image before using the tools.")
            self.untoggle_all_tools()
            return

        # Unlock the label lock
        self.label_window.unlock_label_lock()

        action = self.sender()
        if action == self.select_tool_action:
            if state:
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("select")
            else:
                self.toolChanged.emit(None)

        elif action == self.patch_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("patch")
            else:
                self.toolChanged.emit(None)

        elif action == self.rectangle_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("rectangle")
            else:
                self.toolChanged.emit(None)

        elif action == self.polygon_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("polygon")
            else:
                self.toolChanged.emit(None)
                
        elif action == self.brush_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("brush")
            else:
                self.toolChanged.emit(None)

        elif action == self.erase_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("erase")
            else:
                self.toolChanged.emit(None)

        elif action == self.dropper_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("dropper")
            else:
                self.toolChanged.emit(None)

        elif action == self.fill_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("fill")
            else:
                self.toolChanged.emit(None)

        elif action == self.sam_tool_action:
            if not self.sam_deploy_predictor_dialog.loaded_model:
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
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("sam")
            else:
                self.toolChanged.emit(None)

        elif action == self.see_anything_tool_action:
            if not self.see_anything_deploy_predictor_dialog.loaded_model:
                self.see_anything_tool_action.setChecked(False)
                QMessageBox.warning(self,
                                    "See Anything (YOLOE)",
                                    "You must deploy a Predictor before using the tool.")
                return
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("see_anything")
            else:
                self.toolChanged.emit(None)

        elif action == self.work_area_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.scale_tool_action.setChecked(False)

                self.toolChanged.emit("work_area")
            else:
                self.toolChanged.emit(None)
                
        elif action == self.scale_tool_action:
            if state:
                self.select_tool_action.setChecked(False)
                self.patch_tool_action.setChecked(False)
                self.rectangle_tool_action.setChecked(False)
                self.polygon_tool_action.setChecked(False)
                self.brush_tool_action.setChecked(False)
                self.erase_tool_action.setChecked(False)
                self.dropper_tool_action.setChecked(False)
                self.fill_tool_action.setChecked(False)
                self.sam_tool_action.setChecked(False)
                self.see_anything_tool_action.setChecked(False)
                self.work_area_tool_action.setChecked(False)

                self.toolChanged.emit("scale")
            else:
                self.toolChanged.emit(None)

    def untoggle_all_tools(self):
        """Untoggle all tool actions and unlock the label lock."""
        # Unlock the label lock
        self.label_window.unlock_label_lock()

        # Untoggle all tools
        self.select_tool_action.setChecked(False)
        self.patch_tool_action.setChecked(False)
        self.rectangle_tool_action.setChecked(False)
        self.polygon_tool_action.setChecked(False)
        self.brush_tool_action.setChecked(False)
        self.erase_tool_action.setChecked(False)
        self.dropper_tool_action.setChecked(False)
        self.fill_tool_action.setChecked(False)
        self.sam_tool_action.setChecked(False)
        self.see_anything_tool_action.setChecked(False)
        self.work_area_tool_action.setChecked(False)
        self.scale_tool_action.setChecked(False)

        # Emit to reset the tool
        self.toolChanged.emit(None)

    def handle_tool_changed(self, tool):
        """Update the toolbar UI to reflect the currently selected tool."""
        # Unlock the label lock
        self.label_window.unlock_label_lock()

        if tool == "select":
            self.select_tool_action.setChecked(True)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "patch":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(True)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "rectangle":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(True)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "polygon":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(True)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "brush":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(True)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "erase":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(True)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "dropper":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(True)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "fill":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(True)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "sam":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(True)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "see_anything":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(True)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)

        elif tool == "work_area":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(True)
            self.scale_tool_action.setChecked(False)
            
        elif tool == "scale":
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(True)

        else:
            self.select_tool_action.setChecked(False)
            self.patch_tool_action.setChecked(False)
            self.rectangle_tool_action.setChecked(False)
            self.polygon_tool_action.setChecked(False)
            self.brush_tool_action.setChecked(False)
            self.erase_tool_action.setChecked(False)
            self.dropper_tool_action.setChecked(False)
            self.fill_tool_action.setChecked(False)
            self.sam_tool_action.setChecked(False)
            self.see_anything_tool_action.setChecked(False)
            self.work_area_tool_action.setChecked(False)
            self.scale_tool_action.setChecked(False)
    
    def get_available_devices(self):
        """Get a list of available devices for PyTorch."""
        devices = ['cpu',]
        if torch.backends.mps.is_available():
            devices.append('mps')
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        return devices

    def toggle_device(self):
        """Open a dialog to select the device and update the icon and tooltip accordingly."""
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
        """Handle actions needed when the image is changed."""
        if self.annotation_window.selected_tool == 'sam':
            self.annotation_window.tools['sam'].cancel_working_area()
        if self.annotation_window.selected_tool == 'see_anything':
            self.annotation_window.tools['see_anything'].cancel_working_area()
        
        # Update label tooltips with current counts
        self.label_window.update_tooltips()
            
    def on_image_window_filter_toggled(self, is_expanded):
        """
        Adjusts the vertical stretch between ImageWindow and ConfidenceWindow
        when the filter group in ImageWindow is toggled.
        """
        if is_expanded:
            # Reset to default stretch factors (54 / 46)
            self.image_layout.setStretch(0, 54)  # index 0 is image_window
            self.image_layout.setStretch(1, 46)  # index 1 is confidence_window
        else:
            # Filters are hidden, give less space to image_window
            # and more to confidence_window.
            self.image_layout.setStretch(0, 54)
            self.image_layout.setStretch(1, 66)
            
    def on_image_loaded_check_z_channel(self, image_path):
        """
        Check if the newly loaded image has a z-channel.
        If it doesn't, disable all z-channel UI elements.
        
        Args:
            image_path (str): Path of the loaded image
        """
        raster = self.image_window.raster_manager.get_raster(image_path)
        if raster and raster.z_channel is None:
            # Image has no z-channel, disable UI elements
            self.z_label.setText("Z: -----")
            self.z_label.setEnabled(False)
            self.z_unit_dropdown.setEnabled(False)
            self.z_colormap_dropdown.setEnabled(False)
            self.z_dynamic_button.setEnabled(False)
            self.z_colormap_dropdown.setCurrentText("None")
        elif raster and raster.z_channel is not None:
            # Image has z-channel, enable UI elements
            self.z_label.setEnabled(True)
            self.z_unit_dropdown.setEnabled(True)
            self.z_colormap_dropdown.setEnabled(True)
            # Only enable dynamic button if colormap is not set to "None"
            if self.z_colormap_dropdown.currentText() != "None":
                self.z_dynamic_button.setEnabled(True)
            else:
                self.z_dynamic_button.setEnabled(False)

    def on_z_channel_removed(self, image_path):
        """
        Handle z-channel removal for a raster.
        
        Args:
            image_path (str): Path of the raster with removed z-channel
        """
        # If the removed z-channel belongs to the currently displayed image,
        # clear the z-label in the status bar and disable the dropdown
        if image_path == self.annotation_window.current_image_path:
            self.z_label.setText("Z: -----")
            self.z_label.setEnabled(False)
            self.z_unit_dropdown.setEnabled(False)
            self.z_colormap_dropdown.setEnabled(False)
            self.z_dynamic_button.setEnabled(False)
            self.z_colormap_dropdown.setCurrentText("None")

    def update_project_label(self):
        """Update the project label in the status bar"""

        text = f"CoralNet-Toolbox v{self.version} "
        if self.current_project_path:
            text += f"[Project: {self.current_project_path}]"

        # Update the window title
        self.setWindowTitle(text)

    def update_mouse_position(self, x, y):
        """Update the mouse position label in the status bar"""
        self.mouse_position_label.setText(f"Mouse: X: {x}, Y: {y}")
        
        # Store current mouse position for z-channel lookup
        self.current_mouse_x = x
        self.current_mouse_y = y
        
        # Update z-channel value at new mouse position
        raster = None
        if self.annotation_window.current_image_path:
            raster = self.image_window.raster_manager.get_raster(
                self.annotation_window.current_image_path
            )
        self.update_z_value_at_mouse_position(raster)
        
    def update_image_dimensions(self, width, height):
        """Update the image dimensions label in the status bar"""
        self.image_dimensions_label.setText(f"Image: {height} x {width}")

    def update_view_dimensions(self, original_width, original_height):
        """Update the view dimensions label in the status bar"""
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

        # Update the pixel-based view dimensions
        self.view_dimensions_label.setText(f"View: {height} x {width}")
        
        raster = None
        if self.annotation_window.current_image_path:
            raster = self.image_window.raster_manager.get_raster(
                self.annotation_window.current_image_path
            )

        if raster and raster.scale_units:
            # Scale exists, calculate base meter values
            self.scaled_view_width_m = width * raster.scale_x
            self.scaled_view_height_m = height * raster.scale_y
            
            # Check if the scale unit dropdown was previously disabled
            was_disabled = not self.scale_unit_dropdown.isEnabled()

            # Enable the scale widgets
            self.scaled_view_prefix_label.setEnabled(True)
            self.scaled_view_dims_label.setEnabled(True)
            self.scale_unit_dropdown.setEnabled(True)
            
            # If it was disabled before, set to the last selected unit by default
            if was_disabled:
                self.scale_unit_dropdown.blockSignals(True)
                self.scale_unit_dropdown.setCurrentText(self.current_unit_scale)
                self.scale_unit_dropdown.blockSignals(False)

            # Manually call the update function to display the new values
            self.on_scale_unit_changed(self.scale_unit_dropdown.currentText())

        else:
            # No scale, disable and reset
            self.scaled_view_width_m = 0.0
            self.scaled_view_height_m = 0.0
            
            self.scaled_view_prefix_label.setEnabled(False)
            self.scaled_view_dims_label.setText("0 x 0")
            self.scaled_view_dims_label.setEnabled(False)
            self.scale_unit_dropdown.setEnabled(False)
            
        # Update z_label with z-channel value at current mouse position
        self.update_z_value_at_mouse_position(raster)
    
    def update_z_value_at_mouse_position(self, raster):
        """Update the z_label with z-channel value at current mouse position."""
        if raster and raster.z_channel_lazy is not None:
            # Check if mouse coordinates are within image bounds
            if (0 <= self.current_mouse_x < raster.width and 
                0 <= self.current_mouse_y < raster.height):
                
                try:
                    # Get z-channel value at mouse position (stored as height, width array)
                    z_channel = raster.z_channel_lazy
                    z_value = z_channel[int(self.current_mouse_y), int(self.current_mouse_x)]
                    
                    # Check if the value is NaN (only possible for float types)
                    # Use try-except to handle both float and integer types safely
                    is_nan = False
                    try:
                        is_nan = np.isnan(z_value)
                    except (TypeError, ValueError):
                        # isnan() fails on integer types, which is expected
                        is_nan = False
                    
                    # Check if the value matches the nodata value
                    is_nodata = (raster.z_nodata is not None and float(z_value) == float(raster.z_nodata))
                    
                    if is_nan or is_nodata:
                        self.z_label.setText("Z: ----")
                    else:
                        # Cache the raw z-value for re-conversion if unit dropdown changes
                        self.current_z_value = z_value
                        
                        # Get the original unit from the raster
                        original_unit = raster.z_unit if raster.z_unit else 'm'
                        
                        # Convert to selected unit if different from original
                        display_value = z_value
                        if self.current_unit_z != original_unit:
                            display_value = convert_scale_units(z_value, original_unit, self.current_unit_z)
                        
                        # Format the display based on data type
                        if z_channel.dtype == np.float32:
                            self.z_label.setText(f"Z: {display_value:.3f}")
                        else:
                            self.z_label.setText(f"Z: {int(display_value)}")
                    
                    # Enable the z_label and dropdown since we have valid data
                    self.z_label.setEnabled(True)
                    self.z_unit_dropdown.setEnabled(True)
                    self.z_colormap_dropdown.setEnabled(True)
                    # Only enable dynamic button if colormap is not set to "None"
                    if self.z_colormap_dropdown.currentText() != "None":
                        self.z_dynamic_button.setEnabled(True)
                    
                except (IndexError, ValueError):
                    pass
            
    def on_scale_unit_changed(self, to_unit):
        """
        Converts stored meter values to the selected unit and updates the label.
        """
        if not self.scale_unit_dropdown.isEnabled():
            self.scaled_view_dims_label.setText("0 x 0")
            return

        # Convert the stored meter values
        converted_height = convert_scale_units(self.scaled_view_height_m, 'm', to_unit)
        converted_width = convert_scale_units(self.scaled_view_width_m, 'm', to_unit)

        # Update the dimensions label
        self.scaled_view_dims_label.setText(f"{converted_height:.2f} x {converted_width:.2f}")

        # Remember the selected unit
        self.current_unit_scale = to_unit
        
        # Refresh the confidence window if an annotation is selected
        # This is the only refresh needed, as it's the only
        # change that can happen *while* an annotation is displayed.
        if self.confidence_window.annotation:
            self.confidence_window.refresh_display()
    
    def on_z_unit_changed(self, selected_unit):
        """Handle z-unit dropdown changes by re-displaying cached z-value in new unit."""
        # Update the selected unit
        self.current_unit_z = selected_unit
        
        # Re-convert and display the cached z-value in the new unit
        if self.current_z_value is not None:
            try:
                # Get the current raster to fetch original unit and data type info
                raster = self.image_window.raster_manager.get_raster(self.image_window.selected_image_path)
                if raster and raster.z_channel_lazy is not None:
                    original_unit = raster.z_unit if raster.z_unit else 'm'
                    z_channel = raster.z_channel_lazy
                    
                    # Convert from original unit to selected unit
                    converted_value = convert_scale_units(
                        self.current_z_value, 
                        original_unit, 
                        selected_unit
                    )
                    
                    # Format the display based on data type
                    if z_channel.dtype == np.float32:
                        self.z_label.setText(f"Z: {converted_value:.3f}")
                    else:
                        self.z_label.setText(f"Z: {int(converted_value)}")
            except Exception:
                pass  # If conversion fails, keep last value displayed
        
        # Refresh the confidence window if an annotation is selected
        if self.confidence_window.annotation:
            self.confidence_window.refresh_display()
        
    def on_z_colormap_changed(self, colormap_name):
        """Handle z-colormap dropdown changes by updating the annotation window."""
        self.annotation_window.update_z_colormap(colormap_name)
        
        # Disable the dynamic range button if colormap is set to "None"
        if colormap_name == "None":
            self.z_dynamic_button.setEnabled(False)
            self.z_dynamic_button.setChecked(False)
        else:
            # Enable the dynamic range button if a valid colormap is selected and Z data is available
            if self.annotation_window.z_data_raw is not None:
                self.z_dynamic_button.setEnabled(True)
    
    def on_z_dynamic_toggled(self, checked):
        """Handle z-dynamic scaling button toggle."""
        self.annotation_window.toggle_dynamic_z_scaling(checked)
        
    def get_transparency_value(self):
        """Get the current transparency value from the slider"""
        return self.transparency_slider.value()

    def update_transparency_slider(self, transparency):
        """"Update the transparency slider value"""
        self.transparency_slider.setValue(transparency)

    def update_label_transparency(self, value):
        """Update the transparency for all annotations in the current image."""
        # Make cursor busy
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # Clamp the transparency value to valid range
        transparency = max(0, min(255, value))
        
        # Update transparency slider position
        if self.transparency_slider.value() != transparency:
            # Temporarily block signals to prevent infinite recursion
            self.transparency_slider.blockSignals(True)
            self.transparency_slider.setValue(transparency)
            self.transparency_slider.blockSignals(False)

        # Update transparency for ALL vector annotations in the current image
        # (regardless of visibility - this ensures hidden annotations have correct transparency when shown)
        for annotation in self.annotation_window.get_image_annotations():
            annotation.update_transparency(transparency)

        try:
            # Handle mask annotation updates
            mask = self.annotation_window.current_mask_annotation
            if mask:
                self.label_window.set_mask_transparency(transparency)
        except Exception as e:
            pass

        # Restore cursor
        QApplication.restoreOverrideCursor()
    
    def get_max_detections(self):
        """Get the current max detections value"""
        return self.max_detections
    
    def update_max_detections(self, value):
        """Update the max detections value"""
        if self.max_detections != value:
            self.max_detections = value
            self.max_detections_spinbox.setValue(self.max_detections)
            self.maxDetectionsChanged.emit(value)
        
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
        self.update_uncertainty_thresh(self.uncertainty_thresh)

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
        self.update_iou_thresh(self.iou_thresh)

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
            self.area_threshold_min_slider.setValue(int(min_val * 100))
            self.area_threshold_max_slider.setValue(int(max_val * 100))
            self.areaChanged.emit(min_val, max_val)

    def update_area_label(self):
        """Handle changes to area threshold range slider"""
        min_val = self.area_threshold_min_slider.value()
        max_val = self.area_threshold_max_slider.value()
        if min_val > max_val:
            min_val = max_val
            self.area_threshold_min_slider.setValue(min_val)
        self.area_thresh_min = min_val / 100.0
        self.area_thresh_max = max_val / 100.0
        self.area_threshold_label.setText(f"{self.area_thresh_min:.2f} - {self.area_thresh_max:.2f}")
        self.update_area_thresh(self.area_thresh_min, self.area_thresh_max)

    def open_new_project(self):
        """Confirm user wants to create a new project before closing window."""
        reply = QMessageBox.question(self, "New Project",
                                     "Are you sure you want to create a new project?\n\n"
                                     "All unsaved data will be deleted.",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            
            app = QApplication.instance()
            app.setQuitOnLastWindowClosed(False)
            
            self.close()  # This cleans up the current window
            new_window = MainWindow(self.version)
            new_window.show()

            app.setQuitOnLastWindowClosed(True)

    def open_open_project_dialog(self):
        """Open the Open Project dialog to select a project directory"""
        try:
            self.untoggle_all_tools()
            self.open_project_dialog.exec_()

            # Update the current project path
            path = self.open_project_dialog.get_project_path()
            if path:
                self.current_project_path = path
                self.update_project_label()

        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_save_project_dialog(self):
        """Open the Save Project dialog to select a project directory"""
        try:
            self.untoggle_all_tools()
            self.save_project_dialog.exec_()

            # Update the current project path
            path = self.save_project_dialog.get_project_path()
            if path:
                self.current_project_path = path
                self.update_project_label()

        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def save_project_as(self):
        """Save the project data to a new directory"""
        if self.current_project_path == "":
            self.open_save_project_dialog()
        else:
            self.save_project_dialog.save_project_data(self.current_project_path)

    # TODO update IO classes to have dialogs
    def open_import_frames_dialog(self):
        """Open the Import Frames dialog to import frames into the project"""
        try:
            self.untoggle_all_tools()
            self.import_frames_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_import_viscore_annotations_dialog(self):
        """Open the Import Viscore Annotations dialog to import annotations"""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            self.untoggle_all_tools()
            self.import_viscore_annotations_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_export_viscore_annotations_dialog(self):
        """Open the Export Viscore Annotations dialog to export annotations"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        # Check if there are annotations
        if not self.annotation_window.annotations_dict:
            QMessageBox.warning(self,
                                "Export Annotations",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.export_viscore_annotations_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_export_geojson_annotations_dialog(self):
        """Open the Export GeoJSON dialog to export annotations as GeoJSON files"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        # Check if there are annotations
        if not self.annotation_window.annotations_dict:
            QMessageBox.warning(self,
                                "Export Annotations",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.export_geojson_annotations_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_export_mask_annotations_dialog(self):
        """Open the Export Mask Annotations dialog to export segmentation masks"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        # Check if there are any annotations
        if not self.annotation_window.annotations_dict:
            QMessageBox.warning(self,
                                "Export Segmentation Masks",
                                "No annotations are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.export_mask_annotations_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_depth_dialog(self):
        """Open the Depth Dialog"""
        # TODO: Implement depth dialog functionality
        pass

    def open_explorer_window(self):
        """Open the Explorer window, moving the LabelWindow into it."""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before opening Explorer.")
            return

        # Check if there are any annotations
        if not self.annotation_window.annotations_dict:
            QMessageBox.warning(self,
                                "Explorer",
                                "No annotations are present in the project.")
            return
        
        try:
            self.untoggle_all_tools()

            # Recreate the explorer window, passing the main window instance
            self.explorer_window = ExplorerWindow(self)
            
            # Move the label_window from the main layout to the explorer
            self.label_layout.removeWidget(self.label_window)
            self.label_window.setParent(self.explorer_window.left_panel)  # Re-parent
            self.explorer_window.label_layout.insertWidget(1, self.label_window)  # Add to explorer layout
            
            # Add a spacer to push the timer to the bottom of the left panel while explorer is open
            self.explorer_spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
            self.label_layout.insertItem(0, self.explorer_spacer)
                
            # Disable all main window widgets except select few
            self.set_main_window_enabled_state(
                enable_list=[self.annotation_window, 
                             self.label_window,
                             self.transparency_widget],
                disable_list=[self.toolbar, 
                              self.menu_bar, 
                              self.image_window, 
                              self.confidence_window]
            )
            
            self.explorer_window.showMaximized()
            self.explorer_window.activateWindow()
            self.explorer_window.raise_()
            
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            if self.explorer_window:
                self.explorer_window.close()  # Ensure cleanup
                
            self.explorer_window = None
            # Re-enable everything if there was an error
            self.set_main_window_enabled_state()
        
    def explorer_closed(self):
        """Handle the explorer window being closed."""
        if self.explorer_window:
            # Remove the spacer that was added when explorer opened
            if hasattr(self, 'explorer_spacer'):
                self.label_layout.removeItem(self.explorer_spacer)
                del self.explorer_spacer
            
            # Move the label_window back to the main window's layout
            self.label_window.setParent(self.central_widget)  # Re-parent back
            # Insert at index 0 to maintain original order: label_window first, timer_group second
            self.label_layout.insertWidget(0, self.label_window)
            self.label_window.show()
            self.label_window.resizeEvent(None)
            self.resizeEvent(None)
            
            # Re-enable all main window widgets
            self.set_main_window_enabled_state()
            
            # Clean up reference
            self.explorer_window = None

    def open_patch_annotation_sampling_dialog(self):
        """Open the Patch Annotation Sampling dialog to sample annotations from images"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Proceed to open the dialog if images are loaded
            self.untoggle_all_tools()
            self.patch_annotation_sampling_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_coralnet_authenticate_dialog(self):
        """Open the CoralNet Authenticate dialog to authenticate with CoralNet"""
        try:
            self.untoggle_all_tools()
            self.coralnet_authenticate_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_coralnet_download_dialog(self):
        """Open the CoralNet Download dialog to download datasets from CoralNet"""
        if not self.coralnet_authenticate_dialog.authenticated:
            QMessageBox.warning(self,
                                "CoralNet Download",
                                "You must authenticate with CoralNet before downloading datasets.")
            return
        try:
            self.untoggle_all_tools()
            self.coralnet_download_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_import_dataset_dialog(self):
        """Open the Import Dataset dialog to import datasets into the project."""
        try:
            self.untoggle_all_tools()
            self.import_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_export_dataset_dialog(self):
        """Open the Classify Export Dataset dialog to export classification datasets."""
        # Check if there are loaded images
        if not self.image_window.raster_manager.image_paths:
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
        """Open the Detect Export Dataset dialog to export detection datasets."""
        # Check if there are loaded images
        if not self.image_window.raster_manager.image_paths:
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
        """Open the Segment Export Dataset dialog to export segmentation datasets."""
        # Check if there are loaded images
        if not self.image_window.raster_manager.image_paths:
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
            
    def open_semantic_export_dataset_dialog(self):
        """Open the Semantic Export Dataset dialog to export semantic segmentation datasets."""
        # Check if there are loaded images
        if not self.image_window.raster_manager.image_paths:
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
            self.semantic_export_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_merge_datasets_dialog(self):
        """Open the Classify Merge Datasets dialog to merge datasets."""
        try:
            self.untoggle_all_tools()
            self.classify_merge_datasets_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_tile_dataset_dialog(self):
        """Open the Classify Tile Dataset dialog to classify tiled images."""
        try:
            self.untoggle_all_tools()
            self.classify_tile_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_tile_dataset_dialog(self):
        """Open the Detect Tile Dataset dialog to detect tiled images."""
        try:
            self.untoggle_all_tools()
            self.detect_tile_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_tile_dataset_dialog(self):
        """Open the Segment Tile Dataset dialog to segment tiled images."""
        try:
            self.untoggle_all_tools()
            self.segment_tile_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_semantic_tile_dataset_dialog(self):
        """Open the Semantic Tile Dataset dialog to perform semantic segmentation on tiled images."""
        try:
            self.untoggle_all_tools()
            self.semantic_tile_dataset_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_tile_manager_dialog(self):
        """Open the Tile Manager dialog to create work areas on images."""
        # Check if there are loaded images
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "Tile Inference",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.tile_manager_dialog.show()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_classify_tune_model_dialog(self):
        """Open the Classify Tune Model dialog to tune a classification model."""
        try:
            self.untoggle_all_tools()
            self.classify_tune_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_detect_tune_model_dialog(self):
        """Open the Detect Tune Model dialog to tune a detection model."""
        try:
            self.untoggle_all_tools()
            self.detect_tune_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_segment_tune_model_dialog(self):
        """Open the Segment Tune Model dialog to tune a segmentation model."""
        try:
            self.untoggle_all_tools()
            self.segment_tune_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_semantic_tune_model_dialog(self):
        """Open the Semantic Tune Model dialog to tune a semantic segmentation model."""
        try:
            self.untoggle_all_tools()
            self.semantic_tune_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_train_model_dialog(self):
        """Open the Classify Train Model dialog to train a classification model."""
        try:
            self.untoggle_all_tools()
            self.classify_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_train_model_dialog(self):
        """Open the Detect Train Model dialog to train a detection model."""
        try:
            self.untoggle_all_tools()
            self.detect_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_train_model_dialog(self):
        """Open the Segment Train Model dialog to train a segmentation model."""
        try:
            self.untoggle_all_tools()
            self.segment_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_semantic_train_model_dialog(self):
        """Open the Semantic Train Model dialog to train a semantic segmentation model."""
        try:
            self.untoggle_all_tools()
            self.semantic_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_evaluate_model_dialog(self):
        """Open the Classify Evaluate Model dialog to evaluate a classification model."""
        try:
            self.untoggle_all_tools()
            self.classify_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_detect_evaluate_model_dialog(self):
        """Open the Detect Evaluate Model dialog to evaluate a detection model."""
        try:
            self.untoggle_all_tools()
            self.detect_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_segment_evaluate_model_dialog(self):
        """Open the Segment Evaluate Model dialog to evaluate a segmentation model."""
        try:
            self.untoggle_all_tools()
            self.segment_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_semantic_evaluate_model_dialog(self):
        """Open the Semantic Evaluate Model dialog to evaluate a semantic segmentation model."""
        try:
            self.untoggle_all_tools()
            self.semantic_evaluate_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_optimize_model_dialog(self):
        """Open the Optimize Model dialog to optimize a model."""
        # Check if TensorRT is available
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            QMessageBox.warning(self,
                                "TensorRT Not Installed",
                                "TensorRT is not installed. Some optimization features may not be available. "
                                "Please install TensorRT for full functionality (not available on MacOS).")
        
        # Check if the user has a GPU available using torch
        if not torch.cuda.is_available():
            QMessageBox.warning(self,
                                "Optimize Model",
                                "A GPU is required to optimize models. No GPU was detected.")
            return
        
        try:
            self.untoggle_all_tools()
            self.optimize_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_classify_deploy_model_dialog(self):
        """Open the Classify Deploy Model dialog to deploy a classification model."""
        if not self.image_window.raster_manager.image_paths:
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
        """Open the Detect Deploy Model dialog to deploy a detection model."""
        if not self.image_window.raster_manager.image_paths:
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
        """Open the Segment Deploy Model dialog to deploy a segmentation model."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "Segment Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.segment_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_semantic_deploy_model_dialog(self):
        """Open the Semantic Deploy Model dialog to deploy a semantic segmentation model."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "Semantic Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.semantic_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_detect_video_inference_dialog(self):
        """Open the Detect Video Inference dialog to run inference on video files."""
        try:
            self.untoggle_all_tools()
            self.detect_video_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_segment_video_inference_dialog(self):
        """Open the Segment Video Inference dialog to run inference on video files."""
        try:
            self.untoggle_all_tools()
            self.segment_video_inference_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_transformers_deploy_model_dialog(self):
        """Open the Transformers Deploy Model dialog to deploy an Transformers model."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "Transformers Deploy Model",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.transformers_deploy_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_sam_deploy_predictor_dialog(self):
        """Open the SAM Deploy Predictor dialog to deploy a SAM predictor."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Predictor",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_predictor_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_sam_deploy_generator_dialog(self):
        """Open the SAM Deploy Generator dialog to deploy a SAM generator."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "SAM Deploy Generator",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.sam_deploy_generator_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_see_anything_train_model_dialog(self):
        """Open the See Anything Train Model dialog to train a See Anything model."""
        try:
            self.untoggle_all_tools()
            self.see_anything_train_model_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_see_anything_deploy_predictor_dialog(self):
        """Open the See Anything Deploy Predictor dialog to deploy a See Anything predictor."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "See Anything (YOLOE)",
                                "No images are present in the project.")
            return

        try:
            self.untoggle_all_tools()
            self.see_anything_deploy_predictor_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_see_anything_deploy_generator_dialog(self):
        """Open the See Anything Deploy Generator dialog to deploy a See Anything generator."""
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "See Anything (YOLOE)",
                                "No images are present in the project.")
            return
        
        if len(self.label_window.labels) <= 1:
            QMessageBox.warning(self,
                                "See Anything (YOLOE)",
                                "At least one reference label is required for reference.")
            return

        try:
            self.untoggle_all_tools()
            self.see_anything_deploy_generator_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"An error occurred: {e}")
            
    def open_system_monitor_dialog(self):
        """Open the system system monitor window."""
        if self.system_monitor is None or sip.isdeleted(self.system_monitor):
            self.system_monitor = SystemMonitor()
        
        # Show the monitor window
        self.system_monitor.show()
        self.system_monitor.activateWindow()
        self.system_monitor.raise_()
            
    def open_usage_dialog(self):
        """Display QMessageBox with link to create new issue on GitHub."""
        try:
            self.untoggle_all_tools()
            # URL to create a new issue
            here = '<a href="https://jordan-pierce.github.io/CoralNet-Toolbox/usage">here</a>'
            msg = QMessageBox()
            msg.setWindowIcon(self.coral_icon)
            msg.setWindowTitle("Usage Information")
            msg.setText(f'For information on how to use the toolbox, see {here}.')
            msg.setTextFormat(Qt.RichText)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_check_for_updates_dialog(self, on_open=False):
        """
        Checks if package version is up to date with PyPI.
        """
        try:
            # Get package info from PyPI
            response = requests.get("https://pypi.org/pypi/coralnet-toolbox/json", timeout=5)
            response.raise_for_status()

            # Extract latest version
            package_info = response.json()
            latest_version = package_info["info"]["version"]

            # Compare versions
            needs_update = version.parse(latest_version) > version.parse(self.version)

            if needs_update:
                pip_command = "\npip install coralnet-toolbox=={}".format(latest_version)
                # Create a QMessageBox instance
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Information)
                msg_box.setWindowTitle("Hey, there's an update available!")
                # Set the main message text with hyperlink to usage page
                usage_link = '<a href="https://jordan-pierce.github.io/CoralNet-Toolbox/usage">usage page</a>'
                msg_box.setText(
                    f"A new version ({latest_version}) is available.<br><br>"
                    f"To update, run the following command in your terminal:<br>"
                    f"<b>{pip_command}</b><br><br>"
                    f'Be sure to check out the {usage_link} for any changes!'
                )
                msg_box.setTextFormat(Qt.RichText)
                msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse |
                                                Qt.TextSelectableByKeyboard |
                                                Qt.LinksAccessibleByMouse)

                msg_box.setStandardButtons(QMessageBox.Ok)
                # Execute the dialog
                msg_box.exec_()
            else:
                if not on_open:
                    QMessageBox.information(self,
                                            "Nope, you're good!",
                                            f"You are using the most current version ({self.version}).")

        except (requests.RequestException, KeyError, ValueError) as e:
            if not on_open:
                QMessageBox.warning(self,
                                    "Update Check Failed",
                                    f"Could not check for updates.\nError: {e}")

    def open_create_new_issue_dialog(self):
        """Display QMessageBox with link to create new issue on GitHub."""
        try:
            self.untoggle_all_tools()
            # URL to create a new issue
            here = '<a href="https://github.com/Jordan-Pierce/CoralNet-Toolbox/issues/new/choose">here</a>'
            msg = QMessageBox()
            msg.setWindowIcon(self.coral_icon)
            msg.setWindowTitle("Issues / Feature Requests")
            msg.setText(f'Click {here} to create a new issue or feature request.')
            msg.setTextFormat(Qt.RichText)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def check_windows_gdi_count(self):
        """Calculate and print the number of GDI objects for the current process on Windows."""        
        # 1. Check if the OS is Windows. If not, return early.
        if os.name != 'nt':
            return
        
        # Load necessary libraries
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        user32 = ctypes.WinDLL('user32', use_last_error=True)

        # Define constants
        PROCESS_QUERY_INFORMATION = 0x0400
        GR_GDIOBJECTS = 0

        process_handle = None
        try:
            # 2. Get a handle to the process from its PID
            process_handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, self.pid)
            
            if not process_handle:
                error_code = ctypes.get_last_error()
                raise RuntimeError(f"Failed to open PID {self.pid}. Error code: {error_code}")

            # 3. Use the handle to get the GDI object count
            gdi_count = user32.GetGuiResources(process_handle, GR_GDIOBJECTS)
            
            if gdi_count >= 9500:  # GDI limit
                self.show_gdi_limit_warning()

        except Exception as e:
            pass

        finally:
            # 4. CRITICAL: Always close the handle when you're done
            if process_handle:
                kernel32.CloseHandle(process_handle)
                
        return
            
    def show_gdi_limit_warning(self):
        """
        Show a warning dialog if the GDI limit is reached.
        """
        try:
            self.untoggle_all_tools()
            msg = QMessageBox()
            msg.setWindowIcon(self.coral_icon)
            msg.setWindowTitle("GDI Limit Reached")
            msg.setText(
                "The GDI limit is getting dangerously close to being reached (this is a known issue). "
                "Please immediately save your progress, close, and re-open the application. Failure to do so may "
                "result in data loss."
            )
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_snake_game_dialog(self):
        """
        Open the QtSnakeGame in a new window.
        """
        try:
            self.untoggle_all_tools()
            self.snake_game_dialog.start_game()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")
            
    def open_breakout_game_dialog(self):
        """
        Open the QtBreakoutGame in a new window.
        """
        try:
            self.untoggle_all_tools()
            self.breakout_game_dialog.start_game()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_light_cycle_game_dialog(self):
        """
        Open the QtLightCycleGame in a new window.
        """
        try:
            self.untoggle_all_tools()
            self.lightcycle_game_dialog.start_game()
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
        self.toggle_action.setCheckable(False)
        self.toggle_action.triggered.connect(self.toggle_content)

        # Header button using the action
        self.toggle_button = QToolButton()
        self.toggle_button.setDefaultAction(self.toggle_action)
        self.toggle_button.setCheckable(False)
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

    def toggle_content(self):
        if self.popup.isVisible():
            self.popup.hide()
        else:
            # Position popup below and to the left of the button
            pos = self.toggle_button.mapToGlobal(QPoint(0, 0))
            popup_width = self.popup.sizeHint().width()
            self.popup.move(pos.x() - popup_width + self.toggle_button.width(),
                            pos.y() + self.toggle_button.height())
            self.popup.show()

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