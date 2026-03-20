import warnings

import os
import re
import requests

from packaging import version

import torch

# Important, order this way
import PyQt5.QtCore
import PyQtAds
from PyQtAds import ads

from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QSize
from PyQt5.QtWidgets import (QMainWindow, QApplication, QToolBar, QAction, QSizePolicy,
                             QMessageBox, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
                             QSpinBox, QSlider, QDialog, QPushButton, QListWidget)

# Utilities
from coralnet_toolbox.QtEventFilter import GlobalEventFilter
from coralnet_toolbox.QtAnimationManager import AnimationManager
from coralnet_toolbox.QtPerformanceWindow import PerformanceWindow
from coralnet_toolbox.QtTimerWindow import TimerWindow
from coralnet_toolbox.QtLayoutManager import QtLayoutManager

# Main Windows
from coralnet_toolbox.QtAnnotationWindow import AnnotationWindow
from coralnet_toolbox.QtConfidenceWindow import ConfidenceWindow
from coralnet_toolbox.QtImageWindow import ImageWindow
from coralnet_toolbox.QtLabelWindow import LabelWindow

# Explorer Windows
from coralnet_toolbox.Explorer import AnnotationViewerWindow
from coralnet_toolbox.Explorer import EmbeddingViewerWindow
from coralnet_toolbox.Explorer import SelectionManager

# MVAT Windows
from coralnet_toolbox.MVAT import MVATViewer
from coralnet_toolbox.MVAT import CameraGrid
from coralnet_toolbox.MVAT import MVATManager

# Other Dialogs
from coralnet_toolbox.QtBatchInference import BatchInferenceDialog
from coralnet_toolbox.WorkArea import WorkAreaManager as WorkAreaManagerDialog

# Import Dialogs
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
    ImportMaskAnnotations,
    ExportLabels,
    ExportTagLabLabels,
    ExportAnnotations,
    ExportMaskAnnotations,
    ExportGeoJSONAnnotations,
    ExportCoralNetAnnotations,
    ExportViscoreAnnotations,
    ExportTagLabAnnotations,
    ExportSpatialMetrics,
    OpenProject,
    SaveProject
)

# Machine learning dialogs
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
    Optimize as OptimizeModelDialog,
    TileClassifyDataset as ClassifyTileDatasetDialog,
    TileDetectDataset as DetectTileDatasetDialog,
    TileSegmentDataset as SegmentTileDatasetDialog,
    TileSemanticDataset as SemanticTileDatasetDialog,
)

# SAM dialogs
from coralnet_toolbox.SAM import (
    DeployPredictorDialog as SAMDeployPredictorDialog,
    DeployGeneratorDialog as SAMDeployGeneratorDialog,
)

# See Anything dialogs
from coralnet_toolbox.SeeAnything import (
    TrainModelDialog as SeeAnythingTrainModelDialog,
    DeployPredictorDialog as SeeAnythingDeployPredictorDialog,
    DeployGeneratorDialog as SeeAnythingDeployGeneratorDialog,
)

# CoralNet dialogs
from coralnet_toolbox.CoralNet import (
    AuthenticateDialog as CoralNetAuthenticateDialog,
    DownloadDialog as CoralNetDownloadDialog
)

from coralnet_toolbox.QtDockWrapper import DockWrapper

from coralnet_toolbox.Common import (
    CollapsibleSection,
)

# Game dialogs
from coralnet_toolbox.BreakTime import (
    SnakeGame,
    BreakoutGame,
    LightCycleGame
)

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
        self.coralnet_icon = get_icon("coralnet.svg")
        self.coral_icon = get_icon("coral.svg")
        self.select_icon = get_icon("select.svg")
        self.patch_icon = get_icon("patch.svg")
        self.rectangle_icon = get_icon("rectangle.svg")
        self.polygon_icon = get_icon("polygon.svg")
        self.brush_icon = get_icon("brush.svg")
        self.erase_icon = get_icon("erase.svg")
        self.dropper_icon = get_icon("dropper.svg")
        self.fill_icon = get_icon("fill.svg")
        self.sam_icon = get_icon("wizard.svg")
        self.see_anything_icon = get_icon("eye.svg")
        self.tile_icon = get_icon("tile.svg")
        self.workarea_icon = get_icon("workarea.svg")
        self.scale_icon = get_icon("scale.svg")
        self.turtle_icon = get_icon("turtle.svg")
        self.rabbit_icon = get_icon("rabbit.svg")
        self.rocket_icon = get_icon("rocket.svg")
        self.apple_icon = get_icon("apple.svg")
        self.transparent_icon = get_icon("transparent.svg")
        self.opaque_icon = get_icon("opaque.svg")
        self.z_icon = get_icon("z.svg")
        self.dynamic_icon = get_icon("dynamic.svg")
        self.parameters_icon = get_icon("parameters.svg")
        self.add_icon = get_icon("add.svg")
        self.remove_icon = get_icon("remove.svg")
        self.edit_icon = get_icon("edit.svg")
        self.lock_icon = get_icon("lock.svg")
        self.unlock_icon = get_icon("unlock.svg")
        self.home_icon = get_icon("home.svg")

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
        
        # Set the default uncertainty threshold and IoU threshold
        self.max_detections = 500
        self.iou_thresh = 0.50
        self.uncertainty_thresh = 0.20
        self.area_thresh_min = 0.00
        self.area_thresh_max = 0.70

        # Create main windows
        self.annotation_window = AnnotationWindow(self)
        self.image_window = ImageWindow(self)
        self.label_window = LabelWindow(self)
        self.confidence_window = ConfidenceWindow(self)     
        self.timer_window = TimerWindow(self)   
        self.performance_window = PerformanceWindow(self) 
         
        # Create dock-based mvat windows
        self.mvat_viewer = MVATViewer(self)
        self.camera_grid = CameraGrid(model=None, mvat_window=None)
        self.mvat_manager = MVATManager(self, self.mvat_viewer, self.camera_grid)
        self.camera_grid.model = self.mvat_manager.selection_model
        # Wire a reference to the main window so CameraGrid can access the
        # MVAT manager (used by the Load Cameras toolbar button).
        try:
            self.camera_grid.mvat_window = self
            # Enable the load button if it exists
            if hasattr(self.camera_grid, 'load_btn'):
                self.camera_grid.load_btn.setEnabled(True)
        except Exception:
            pass
        
        # Create dock-based explorer windows
        self.annotation_viewer_window = AnnotationViewerWindow(self)
        self.annotation_viewer_window.set_animation_manager(self.animation_manager)
        self.embedding_viewer_window = EmbeddingViewerWindow(self)
        self.embedding_viewer_window.set_animation_manager(self.animation_manager)
        self.annotation_viewer_window.cleared.connect(self.embedding_viewer_window.clear_view)

        # Create the centralized selection manager for explorer windows
        self.selection_manager = SelectionManager(self)
        self.selection_manager.register_annotation_viewer(self.annotation_viewer_window)
        self.selection_manager.register_embedding_viewer(self.embedding_viewer_window)
        self.selection_manager.register_annotation_window(self.annotation_window)
        self.selection_manager.register_label_window(self.label_window)
        self.selection_manager.register_confidence_window(self.confidence_window)
        
        # Initialize after other windows are created
        self.annotation_window.initialize_tools()

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
        self.import_mask_annotations_dialog = ImportMaskAnnotations(self)
        self.export_labels = ExportLabels(self)
        self.export_taglab_labels = ExportTagLabLabels(self)
        self.export_annotations = ExportAnnotations(self)
        self.export_coralnet_annotations = ExportCoralNetAnnotations(self)
        self.export_viscore_annotations_dialog = ExportViscoreAnnotations(self)
        self.export_taglab_annotations = ExportTagLabAnnotations(self)
        self.export_mask_annotations_dialog = ExportMaskAnnotations(self)
        self.export_geojson_annotations_dialog = ExportGeoJSONAnnotations(self)
        self.export_spatial_metrics_dialog = ExportSpatialMetrics(self)
        self.import_frames_dialog = ImportFrames(self)
        self.open_project_dialog = OpenProject(self)
        self.save_project_dialog = SaveProject(self)

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

        # Create dialogs (Batch Inference - Consolidated)
        # This is accessed via ImageWindow right-click context menu
        self.batch_inference_dialog = BatchInferenceDialog(self)

        # Create dialogs (Work Areas)
        self.tile_manager_dialog = WorkAreaManagerDialog(self)
        self.classify_tile_dataset_dialog = ClassifyTileDatasetDialog(self)
        self.detect_tile_dataset_dialog = DetectTileDatasetDialog(self)
        self.segment_tile_dataset_dialog = SegmentTileDatasetDialog(self)
        self.semantic_tile_dataset_dialog = SemanticTileDatasetDialog(self)

        # Create dialogs (Break Time)
        self.snake_game_dialog = SnakeGame(self)
        self.breakout_game_dialog = BreakoutGame(self)
        self.lightcycle_game_dialog = LightCycleGame(self)

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
        # Import Mask Annotations
        self.import_mask_annotations_action = QAction("Masks (PNG)", self)
        self.import_mask_annotations_action.triggered.connect(self.open_import_mask_annotations_dialog)
        self.import_annotations_menu.addAction(self.import_mask_annotations_action)

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
        self.export_mask_annotations_action = QAction("Masks (PNG)", self)
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
        self.export_menu.addSeparator()
        
        # Export Spatial Metrics (at Export menu level, not in Annotations submenu)
        self.export_spatial_metrics_action = QAction("Spatial Metrics", self)
        self.export_spatial_metrics_action.triggered.connect(self.export_spatial_metrics_dialog.exec_)
        self.export_menu.addAction(self.export_spatial_metrics_action)
        
        # Add a separator
        self.file_menu.addSeparator()

        # New Project
        self.new_project_action = QAction("New Project", self)
        self.new_project_action.triggered.connect(self.open_new_project)
        self.file_menu.addAction(self.new_project_action)
        # Open Project
        self.open_project_action = QAction("Open Project", self)
        self.open_project_action.triggered.connect(self.open_open_project_dialog)
        self.file_menu.addAction(self.open_project_action)
        # Save Project
        self.save_project_action = QAction("Save Project", self)
        self.save_project_action.setToolTip("Ctrl + Shift + S")
        self.save_project_action.triggered.connect(self.open_save_project_dialog)
        self.file_menu.addAction(self.save_project_action)
        
        # ========== LAYOUT MENU ==========
        # Layout menu - for saving and loading dock configurations
        self.layout_menu = self.menu_bar.addMenu("Layout")
        self.dock_toggle_actions = {}
        
        # ========== VIEW MENU ==========
        # Fetch the fully encapsulated menu from the MVAT Viewer and add it to the main menu bar
        self.view_menu = self.mvat_viewer.create_view_menu()
        self.menu_bar.addMenu(self.view_menu)

        # ========== UTILITIES MENU ==========
        # Utilities menu
        self.utilities_menu = self.menu_bar.addMenu("Utilities")
        
        # Sampling Annotations
        self.annotation_sampling_action = QAction("Sample Patches", self)
        self.annotation_sampling_action.triggered.connect(self.open_patch_sampling_dialog_dialog)
        self.utilities_menu.addAction(self.annotation_sampling_action)
        
        # Rugosity
        self.rugosity_action = QAction("Measure Rugosity", self)
        self.rugosity_action.triggered.connect(self.open_rugosity_dialog)
        self.utilities_menu.addAction(self.rugosity_action)

        # Scale
        self.scale_action = QAction("Set Scale", self)
        self.scale_action.triggered.connect(self.open_scale_dialog)
        self.utilities_menu.addAction(self.scale_action)

        # Work Area Manager
        self.tile_manager_action = QAction("Work Areas", self)
        self.tile_manager_action.triggered.connect(self.open_tile_manager_dialog)
        self.utilities_menu.addAction(self.tile_manager_action)
        
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

        # ========== MACHINE LEARNING MENU ==========
        # Machine Learning menu
        self.ml_menu = self.menu_bar.addMenu("Machine Learning")

        # Merge Datasets submenu
        self.ml_merge_datasets_menu = self.ml_menu.addMenu("Merge Datasets")
        # Merge Classification Datasets
        self.ml_classify_merge_datasets_action = QAction("Classify", self)
        self.ml_classify_merge_datasets_action.triggered.connect(self.open_classify_merge_datasets_dialog)
        self.ml_merge_datasets_menu.addAction(self.ml_classify_merge_datasets_action)
        
        # Tile Dataset submenu
        self.tile_dataset_menu = self.ml_menu.addMenu("Tile Dataset")
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
        
        # --------------------------------------------------
        # Create collapsible Parameters section
        # --------------------------------------------------        
        self.parameters_section = CollapsibleSection("Parameters", "parameters.svg", position='topright')

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
        
        # Add the parameters section to the toolbar
        self.toolbar.addWidget(self.parameters_section)
        
        self.toolbar.addSeparator()

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

        # --------------------------------------------------
        # 1. Initialize the Advanced Dock Manager
        # --------------------------------------------------
        
        # Instantiate the ADS dock manager.
        self.dock_manager = ads.CDockManager(self)
        
        # Disable the "List all tabs" button in the dock tab bars
        self.dock_manager.setConfigFlags(
            self.dock_manager.configFlags() & ~ads.CDockManager.DockAreaHasTabsMenuButton
        )
        
        # Apply custom QSS for vibrant cyan tabs
        self.dock_manager.setStyleSheet("""
            /* The flat grey background behind the tabs */
            ads--CDockAreaTabBar {
            background-color: #f0f0f0;
            height: 28px;
            }

            /* Inactive tabs - subtle grey, pushed down slightly */
            ads--CDockWidgetTab {
            background-color: #e8e8e8;
            border: none;
            border-right: 1px solid #d0d0d0; /* Subtle separator between inactive tabs */
            padding: 2px 12px;
            color: #666666;
            margin-top: 2px; /* Pushes inactive tabs down so they don't protrude */
            font-weight: 700;
            min-width: 100px;
            max-width: 600px;
            }

            /* Slight highlight when hovering over inactive tabs */
            ads--CDockWidgetTab:hover {
            background-color: #d9d9d9;
            color: #333333;
            }

            /* The currently active tab - Vibrant Cyan, protruding up */
            ads--CDockWidgetTab[activeTab="true"] {
            background-color: #00A8E6; /* Bright, vibrant cyan */
            border: none;
            color: #ffffff; /* Pure white text for maximum contrast */
            font-weight: 800; /* Bold for prominence */
            margin-top: 0px; /* Zero margin pulls it flush to the top, making it protrude */
            padding: 4px 12px;
            min-width: 100px;
            max-width: 600px;
            }

            /* Active tab on hover - slightly darker cyan */
            ads--CDockWidgetTab[activeTab="true"]:hover {
            background-color: #0095CC; /* Slightly darker cyan on hover */
            }

            /* Close button - blend with tab, no button appearance */
            ads--CDockWidgetTab QAbstractButton {
            background-color: transparent;
            border: none;
            padding: 0px;
            margin: 0px 4px;
            width: 14px;
            height: 14px;
            }

            ads--CDockWidgetTab[activeTab="true"] QAbstractButton {
            background-color: transparent;
            }

            ads--CDockWidgetTab QAbstractButton:hover {
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 2px;
            }

            ads--CDockWidgetTab[activeTab="true"] QAbstractButton:hover {
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 2px;
            }

            /* Focus state - keep the same solid block look */
            ads--CDockWidgetTab:focus {
            outline: none; /* Kills default Qt dotted line */
            }
            
            /* Dock area background */
            ads--CDockArea {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            }
        """)

        # --------------------------------------------------
        # 2. Create the Docks & Containers
        # --------------------------------------------------
        
        # Setup the Annotation Dock using DockWrapper
        self.annotation_dock = DockWrapper("Annotation", "AnnotationDock", self.annotation_window, self)
        
        if hasattr(self.annotation_window, 'create_top_toolbar'):
            self.annotation_dock.add_toolbar(self.annotation_window.create_top_toolbar())
        if hasattr(self.annotation_window, 'create_bottom_toolbar'):
            self.annotation_dock.add_toolbar(self.annotation_window.create_bottom_toolbar(), Qt.BottomToolBarArea)

        # Setup Image Dock using DockWrapper
        self.rasters_dock = DockWrapper("Rasters", "RastersDock", self.image_window, self)
        
        if hasattr(self.image_window, 'create_filter_toolbar'):
            self.rasters_dock.add_toolbar(self.image_window.create_filter_toolbar())
        self.rasters_dock.add_toolbar_break()
        if hasattr(self.image_window, 'create_info_toolbar'):
            self.rasters_dock.add_toolbar(self.image_window.create_info_toolbar())
        if hasattr(self.image_window, 'create_action_toolbar'):
            self.rasters_dock.add_toolbar(self.image_window.create_action_toolbar())

        # Setup Label Dock using DockWrapper
        self.labels_dock = DockWrapper("Labels", "LabelsDock",  self.label_window, self)
        
        if hasattr(self.label_window, 'create_action_toolbar'):
            self.labels_dock.add_toolbar(self.label_window.create_action_toolbar())
        self.labels_dock.add_toolbar_break()
        if hasattr(self.label_window, 'create_filter_toolbar'):
            self.labels_dock.add_toolbar(self.label_window.create_filter_toolbar())
        if hasattr(self.label_window, 'create_bottom_toolbar'):
            self.labels_dock.add_toolbar(self.label_window.create_bottom_toolbar(), Qt.BottomToolBarArea)
            
        # Setup Confidence Dock (Right) using DockWrapper
        self.confidence_dock = DockWrapper("Confidence", "ConfidenceDock", self.confidence_window, self)
        
        # Setup Performance Dock (Right) using DockWrapper
        self.performance_dock = DockWrapper("Performance", "PerformanceDock", self.performance_window, self)

        # Setup Timer Dock (Left, below Labels) using DockWrapper
        self.timer_dock = DockWrapper("Timer", "TimerDock", self.timer_window, self)
                
        # Setup Annotation Gallery Dock (Bottom) using DockWrapper
        self.gallery_dock = DockWrapper("Gallery", "GalleryDock", self.annotation_viewer_window, self)
        
        if hasattr(self.annotation_viewer_window, 'create_top_toolbar'):
            self.gallery_dock.add_toolbar(self.annotation_viewer_window.create_top_toolbar())
        if hasattr(self.annotation_viewer_window, 'create_bottom_toolbar'):
            self.gallery_dock.add_toolbar(
                self.annotation_viewer_window.create_bottom_toolbar(),
                Qt.BottomToolBarArea)

        # Setup Embedding Viewer Dock (Bottom) using DockWrapper
        self.embeddings_dock = DockWrapper("Embeddings", "EmbeddingsDock", self.embedding_viewer_window, self)
        
        if hasattr(self.embedding_viewer_window, 'create_top_toolbar'):
            self.embeddings_dock.add_toolbar(self.embedding_viewer_window.create_top_toolbar())
        if hasattr(self.embedding_viewer_window, 'create_bottom_toolbar'):
            self.embeddings_dock.add_toolbar(
                self.embedding_viewer_window.create_bottom_toolbar(),
                Qt.BottomToolBarArea)

        # Setup MVAT Viewer Dock (Bottom-left) using DockWrapper
        self.mvat_dock = DockWrapper("3D Viewer", "3DViewerDock", self.mvat_viewer, self)

        # Setup Camera Grid Dock (Bottom-right) using DockWrapper
        self.grid_dock = DockWrapper("Grid", "GridDock", self.camera_grid, self)
        
        if hasattr(self.camera_grid, 'create_top_toolbar'):
            self.grid_dock.add_toolbar(self.camera_grid.create_top_toolbar())

        # --------------------------------------------------
        # 3. Explicitly arrange the docks using PyQtADS
        # --------------------------------------------------

        # 1. Add Workspace dock first as the central anchor
        annotation_area = self.dock_manager.addDockWidget(ads.TopDockWidgetArea, self.annotation_dock)
        
        # 2. Add Image dock to the right of the Annotation dock
        raster_area = self.dock_manager.addDockWidget(ads.RightDockWidgetArea, self.rasters_dock, annotation_area)
        
        # 3. Add Label dock below the Image dock 
        label_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.labels_dock, raster_area)

        # 3. Add Confidence dock below the Label dock
        conf_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.confidence_dock, label_area)
        
        # 4. Add Performance dock below Confidence
        perf_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.performance_dock, conf_area)

        # 5. TAB the Timer dock into Performance, but hide it initially
        timer_area = self.dock_manager.addDockWidget(ads.CenterDockWidgetArea, self.timer_dock, perf_area)
        
        # 6. Add Annotation Gallery to the Bottom of the WORKSPACE explicitly
        gallery_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.gallery_dock, annotation_area)
        
        # 7. Add Embedding Viewer to the Right of the Annotation Gallery
        embed_area = self.dock_manager.addDockWidget(ads.RightDockWidgetArea, self.embeddings_dock, gallery_area)

        # 8. Place the MVAT Viewer below the Annotation Gallery
        mvat_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.mvat_dock, gallery_area)

        # 9. Place Camera Grid below Embedding Viewer
        grid_area = self.dock_manager.addDockWidget(ads.BottomDockWidgetArea, self.grid_dock, embed_area)
        
        # Populate the Windows menu with dock toggle actions
        dock_windows = [
            ("Annotation", self.annotation_dock),
            ("Rasters", self.rasters_dock),
            ("Labels", self.labels_dock),
            ("Confidence", self.confidence_dock),
            ("Performance", self.performance_dock),
            ("Timer", self.timer_dock),
            ("Gallery", self.gallery_dock),
            ("Embeddings", self.embeddings_dock),
            ("3D Viewer", self.mvat_dock),
            ("Grid", self.grid_dock),
        ]

        for dock_name, dock_widget in dock_windows:
            action = QAction(dock_name, self, checkable=True)
            action.setChecked(dock_widget.isVisible())
            # Connect the action to toggle the dock visibility
            action.triggered.connect(lambda checked, dw=dock_widget: dw.toggleView(checked))
            # Also update the action when the dock visibility changes
            dock_widget.visibilityChanged.connect(lambda visible, act=action: act.setChecked(visible))
            self.layout_menu.addAction(action)
            self.dock_toggle_actions[dock_name] = action

        # Add separator before layout actions
        self.layout_menu.addSeparator()

        # Save Layout action
        self.save_layout_action = QAction("Save Layout", self)
        self.save_layout_action.triggered.connect(self.on_save_layout)
        self.layout_menu.addAction(self.save_layout_action)

        # Load Layout submenu
        self.load_layout_menu = self.layout_menu.addMenu("Load Layout")
        self.populate_load_layout_menu()
        
        # --------------------------------------------------
        # Restore layout from cache
        # --------------------------------------------------
        QtLayoutManager.restore_or_default(
            self.dock_manager,
            layout_name='default'
        )
        
        # --------------------------------------------------
        # Enable drag and drop
        # --------------------------------------------------
        self.setAcceptDrops(True)
        
        # --------------------------------------------------
        # Setup connections
        # --------------------------------------------------

        # ---------------------------------------------------------------------
        # Tool & toolbar synchronization
        # - Emit the chosen tool name (str) to the annotation window so it
        #   can activate the corresponding tool implementation.
        # - Keep the label window's internal "annotation count state" in sync
        #   with the active tool (some tools affect label counts UI).
        # - Ensure toolbar UI updates when the AnnotationWindow requests a
        #   tool change (two-way synchronization).
        # ---------------------------------------------------------------------
        self.toolChanged.connect(self.annotation_window.set_selected_tool)
        self.toolChanged.connect(self.label_window.update_annotation_count_state)
        self.annotation_window.toolChanged.connect(self.handle_tool_changed)

        # ---------------------------------------------------------------------
        # Label and annotation selection flows
        # - When AnnotationWindow reports a label was selected, forward that
        #   selection to the LabelWindow so its UI reflects the selection.
        # - Keep the label-window annotation counts updated when annotations
        #   are selected in the annotation canvas.
        # ---------------------------------------------------------------------
        self.annotation_window.labelSelected.connect(self.label_window.set_selected_label)
        self.annotation_window.annotationSelected.connect(self.label_window.update_annotation_count)

        # ---------------------------------------------------------------------
        # Annotation lifecycle -> Label/Viewer updates
        # - Update label tooltips whenever annotations are created/deleted.
        # - Notify gallery/embedding viewers about annotation creation,
        #   deletion, and modification so they can maintain their datasets.
        # ---------------------------------------------------------------------
        self.annotation_window.annotationCreated.connect(self.label_window.update_tooltips)
        self.annotation_window.annotationDeleted.connect(self.label_window.update_tooltips)

        self.annotation_window.annotationCreated.connect(self.annotation_viewer_window.on_annotation_created)
        self.annotation_window.annotationsCreated.connect(self.annotation_viewer_window.on_annotations_created)
        self.annotation_window.annotationDeleted.connect(self.annotation_viewer_window.on_annotation_deleted)
        self.annotation_window.annotationsDeleted.connect(self.annotation_viewer_window.on_annotations_deleted)
        self.annotation_window.annotationModified.connect(self.annotation_viewer_window.on_annotation_modified)

        self.annotation_window.annotationCreated.connect(self.embedding_viewer_window.on_annotation_created)
        self.annotation_window.annotationsCreated.connect(self.embedding_viewer_window.on_annotations_created)
        self.annotation_window.annotationDeleted.connect(self.embedding_viewer_window.on_annotation_deleted)
        self.annotation_window.annotationsDeleted.connect(self.embedding_viewer_window.on_annotations_deleted)
        self.annotation_window.annotationModified.connect(self.embedding_viewer_window.on_annotation_modified)
        
        # ---------------------------------------------------------------------
        # Annotation label changes -> update viewers
        # - When an annotation's label is changed, both gallery and embedding
        #   viewers need to update any cached label/display state.
        # ---------------------------------------------------------------------
        self.annotation_window.annotationLabelChanged.connect(
            self.annotation_viewer_window.on_annotation_label_changed)
        self.annotation_window.annotationLabelChanged.connect(
            self.embedding_viewer_window.on_annotation_label_changed)
        
        # Batch label changes
        self.annotation_window.annotationsLabelsChanged.connect(
            self.annotation_viewer_window.on_annotations_labels_changed)
        self.annotation_window.annotationsLabelsChanged.connect(
            self.embedding_viewer_window.on_annotations_labels_changed)
        
        # ---------------------------------------------------------------------
        # Annotation geometry changes -> update viewers
        # - When annotations are moved, resized, cut, merged, split, or have
        #   their geometry edited, viewers need to update accordingly.
        # ---------------------------------------------------------------------
        self.annotation_window.annotationMoved.connect(
            self.annotation_viewer_window.on_annotation_moved)
        self.annotation_window.annotationMoved.connect(
            self.embedding_viewer_window.on_annotation_moved)
        
        self.annotation_window.annotationGeometryEdited.connect(
            self.annotation_viewer_window.on_annotation_geometry_edited)
        self.annotation_window.annotationGeometryEdited.connect(
            self.embedding_viewer_window.on_annotation_geometry_edited)
        
        self.annotation_window.annotationCut.connect(
            self.annotation_viewer_window.on_annotation_cut)
        self.annotation_window.annotationCut.connect(
            self.embedding_viewer_window.on_annotation_cut)
        
        self.annotation_window.annotationsMerged.connect(
            self.annotation_viewer_window.on_annotations_merged)
        self.annotation_window.annotationsMerged.connect(
            self.embedding_viewer_window.on_annotations_merged)
        
        self.annotation_window.annotationSplit.connect(
            self.annotation_viewer_window.on_annotation_split)
        self.annotation_window.annotationSplit.connect(
            self.embedding_viewer_window.on_annotation_split)
        
        # ---------------------------------------------------------------------
        # NOTE: Selection synchronization across AnnotationWindow, Viewer
        # Windows and Embedding viewer is handled centrally by
        # SelectionManager (registered earlier). Do not duplicate selection
        # syncing here to avoid inconsistent state.
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Filter -> embedding working set
        # - When the annotation gallery emits a filtered set of annotations,
        #   update the embedding viewer's working set so embeddings reflect
        #   the current gallery filter.
        # ---------------------------------------------------------------------
        self.annotation_viewer_window.annotations_filtered.connect(
            self.embedding_viewer_window.set_working_set)

        # ---------------------------------------------------------------------
        # LabelWindow -> AnnotationWindow flows
        # - Selecting a label in the LabelWindow should change the selected
        #   label in the AnnotationWindow (affects subsequent annotations).
        # - Changing label transparency in the LabelWindow should update
        #   the annotation visualization in the AnnotationWindow.
        # ---------------------------------------------------------------------
        self.label_window.labelSelected.connect(self.annotation_window.set_selected_label)
        self.label_window.transparencyChanged.connect(self.annotation_window.update_label_transparency)

        # ---------------------------------------------------------------------
        # Image-related signals
        # - When the active image selection changes, tell the AnnotationWindow
        #   to update its current image path & state.
        # - When images change, perform higher-level cleanup (e.g. cancel SAM
        #   or YOLOE working areas) via handle_image_changed().
        # ---------------------------------------------------------------------
        self.image_window.imageSelected.connect(self.annotation_window.update_current_image_path)
        self.image_window.imageChanged.connect(self.handle_image_changed)

        # ---------------------------------------------------------------------
        # Z-channel handling
        # - If a Z channel is removed from an image, let the AnnotationWindow
        #   update its status and clear any Z visualizations it was showing.
        # ---------------------------------------------------------------------
        self.image_window.zChannelRemoved.connect(self.annotation_window.on_z_channel_removed)
        self.image_window.zChannelRemoved.connect(self.annotation_window.clear_z_channel_visualization)

        # ---------------------------------------------------------------------
        # Image loaded handlers
        # - When a new image is loaded, several viewers need to refresh or
        #   re-check state (z-channel, gallery filters, embedding context).
        # - Also close any image-specific dialogs/tools that shouldn't persist
        #   across images (patch sampling, rugosity, scale dialogs, etc.).
        # ---------------------------------------------------------------------
        self.image_window.imageLoaded.connect(self.annotation_window.on_image_loaded_check_z_channel)
        self.image_window.imageLoaded.connect(self.annotation_viewer_window.on_image_loaded)
        self.image_window.imageLoaded.connect(self.embedding_viewer_window.on_image_loaded)
        self.annotation_window.imageLoaded.connect(self.close_image_specific_dialogs)
        
        # --------------------------------------------------
        # Setup global event filter for shortcuts
        # --------------------------------------------------
        self.global_event_filter = GlobalEventFilter(self)
        QApplication.instance().installEventFilter(self.global_event_filter)

        # --------------------------------------------------
        # Check for updates on opening
        # --------------------------------------------------
        self.open_check_for_updates_dialog(on_open=True)
        
        # Flash a success message for 3 seconds
        self.status_bar.showMessage("Ready!", 3000)
        
        # Process events
        QApplication.processEvents()
        
    @property
    def status_bar(self):
        """
        Alias for the native QMainWindow status bar. 
        Creates it automatically on the first call if it doesn't exist.
        """
        return self.statusBar()
    
    # --- Backwards Compatibility Aliases for Tools ---
    
    @property
    def current_unit_scale(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.current_unit_scale
        
    @property
    def current_unit_z(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.current_unit_z
        
    def get_transparency_value(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.transparency_slider.value()

    # Redirect widget references for ScaleTool, ZImportDialog, etc.
    @property
    def scale_unit_dropdown(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.scale_unit_dropdown

    @property
    def z_colormap_dropdown(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.z_colormap_dropdown

    @property
    def z_transparency_widget(self):
        if not hasattr(self, 'annotation_window') or self.annotation_window is None:
            return None
        return self.annotation_window.z_transparency_widget
            
    def update_project_label(self):
        """Update the project label in the status bar"""

        text = f"CoralNet-Toolbox v{self.version} "
        if self.current_project_path:
            text += f"[Project: {self.current_project_path}]"

        # Update the window title
        self.setWindowTitle(text)
    
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

    def showEvent(self, event):
        """Show the main window maximized."""
        super().showEvent(event)
        self.showMaximized()

    def closeEvent(self, event):
        """Ensure special windows (explorer, mvat) and Performance Window are closed when the main window closes."""
        
        # Save layout configuration before closing
        if hasattr(self, 'dock_manager'):
            QtLayoutManager.save_and_close(self.dock_manager, layout_name='default')
        
        if hasattr(self, 'mvat_manager') and self.mvat_manager:
            self.mvat_manager.cleanup()
        
        # Stop timer threads properly
        if hasattr(self, 'timer_window') and self.timer_window:
            self.timer_window.stop_threads()
            
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

            # Accept if any of the files is a project file (.json or .bin)
            if any(file_name.lower().endswith(('.json', '.bin')) for file_name in file_names):
                event.acceptProposedAction()
            else:
                self.import_images.dragEnterEvent(event)

    def dropEvent(self, event):
        """Handle drop event for drag-and-drop."""
        self.untoggle_all_tools()

        urls = event.mimeData().urls()
        file_names = [url.toLocalFile() for url in urls if url.isLocalFile()]

        if file_names:
            # Check if a single project file (.json or .bin) was dropped
            if len(file_names) == 1 and file_names[0].lower().endswith(('.json', '.bin')):
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

            # Accept if any of the files is a project file (.json or .bin)
            if any(file_name.lower().endswith(('.json', '.bin')) for file_name in file_names):
                event.acceptProposedAction()
            else:
                self.import_images.dragMoveEvent(event)
                
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
        
    def choose_specific_tool(self, tool, preserve_selection=False):
        """Choose a specific tool based on the provided tool name.
        
        Args:
            tool: The tool name to activate.
            preserve_selection: If True, existing selections will be preserved during tool switch.
        """
        # Update button states to reflect the new tool
        self.handle_tool_changed(tool)
        
        # Set the tool in the annotation window
        self.annotation_window.set_selected_tool(tool, preserve_selection=preserve_selection)
        
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

                self.toolChanged.emit("work_area")
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
                    
        elif tool == "patch_sampling":
            # Patch Sampling has no toolbar button - uncheck all toolbar buttons
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
            
        elif tool == "rugosity":
            # Rugosity has no toolbar button - uncheck all toolbar buttons
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

    def open_import_mask_annotations_dialog(self):
        """Open the Import Mask Annotations dialog to import segmentation masks"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before importing mask annotations.")
            return

        # Check if there are any labels
        if not self.label_window.labels:
            QMessageBox.warning(self,
                                "No Labels",
                                "Please create labels before importing mask annotations.")
            return

        try:
            self.untoggle_all_tools()
            self.import_mask_annotations_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_patch_sampling_dialog_dialog(self):
        """Open the Patch Sampling tool to sample annotations from images"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before sampling annotations.")
            return

        try:
            # Deactivate other tools
            self.untoggle_all_tools()
            # Activate the patch sampling tool
            self.annotation_window.set_selected_tool("patch_sampling")
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_rugosity_dialog(self):
        """Open the Rugosity tool for measuring rugosity and spatial analysis"""
        # Check if there is an image loaded
        image_path = self.annotation_window.current_image_path
        if not image_path:
            QMessageBox.warning(self,
                                "No Image Loaded",
                                "Please load an image before using the Rugosity tool.")
            return
        
        # Check if scale is set on current image
        raster = self.image_window.raster_manager.get_raster(image_path)
        if not raster or not raster.scale_x or not raster.scale_y:
            QMessageBox.warning(self,
                                "Scale Not Set",
                                "The Rugosity tool requires scale to be set on the current image.\n\n"
                                "Please use the Scale Tool to set the scale first.")
            return
        
        # Check the z-channel existing for the current image
        if not raster.has_z_channel():
            QMessageBox.warning(self,
                                "Z-Channel Not Found",
                                "The Rugosity tool requires a z-channel in the current image.\n\n"
                                "Please ensure the image has a z-channel before using the Rugosity tool.")
            return
        
        try:
            # Deactivate other tools
            self.untoggle_all_tools()
            # Activate the rugosity tool
            self.annotation_window.set_selected_tool("rugosity")
        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"{e}")

    def open_scale_dialog(self):
        """Open the Scale tool to set spatial scale on images"""
        # Check if there are any images in the project
        if not self.image_window.raster_manager.image_paths:
            QMessageBox.warning(self,
                                "No Images Loaded",
                                "Please load images into the project before setting scale.")
            return

        try:
            # Deactivate other tools
            self.untoggle_all_tools()
            # Activate the scale tool
            self.annotation_window.set_selected_tool("scale")
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

    def on_save_layout(self):
        """Show save layout dialog and save the current layout."""
        QtLayoutManager.show_save_dialog_and_save(self.dock_manager, parent=self)
        # Refresh the load menu to include the newly saved layout
        self.populate_load_layout_menu()

    def populate_load_layout_menu(self):
        """Discover and populate the Load Layout submenu with available layouts."""
        # Clear existing actions
        self.load_layout_menu.clear()
        
        # Get list of available layouts
        layouts = QtLayoutManager.list_available_layouts()
        
        if not layouts:
            no_layouts_action = QAction("(No saved layouts)", self)
            no_layouts_action.setEnabled(False)
            self.load_layout_menu.addAction(no_layouts_action)
            return
        
        # Add all custom saved layouts
        for layout_name in layouts:
            action = QAction(layout_name, self)
            action.triggered.connect(
                lambda checked, name=layout_name: self.load_specific_layout(name)
            )
            self.load_layout_menu.addAction(action)

    def load_specific_layout(self, layout_name: str):
        """Load a specific layout configuration."""
        success = QtLayoutManager.load_layout(self.dock_manager, layout_name)
        if success:
            QMessageBox.information(
                self,
                "Layout Loaded",
                f"Layout '{layout_name}' has been loaded successfully."
            )
        else:
            QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load layout '{layout_name}'."
            )

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
    
    # Special Windows
    
    def _on_gallery_selection_changed(self, selected_ids):
        """
        DEPRECATED: Selection syncing is now handled by SelectionManager.
        
        This method is kept for backward compatibility but selection
        synchronization is managed centrally by self.selection_manager.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        # SelectionManager handles all selection syncing automatically
        pass
    
    def _on_embedding_selection_changed(self, selected_ids):
        """
        DEPRECATED: Selection syncing is now handled by SelectionManager.
        
        This method is kept for backward compatibility but selection
        synchronization is managed centrally by self.selection_manager.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        # SelectionManager handles all selection syncing automatically
        pass
    
    def _on_annotation_selection_changed(self, selected_ids):
        """
        DEPRECATED: Selection syncing is now handled by SelectionManager.
        
        This method is kept for backward compatibility but selection
        synchronization is managed centrally by self.selection_manager.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        # SelectionManager handles all selection syncing automatically
        pass
            
    def close_image_specific_dialogs(self):
        """Close image-specific dialogs (e.g., patch sampling, rugosity) when a new image is loaded."""
        # Check if there is a dialog tool selected, if so, get the tool
        if self.annotation_window.selected_tool:
            selected_tool = self.annotation_window.selected_tool
            
            # Deactivate scale tool if active
            if selected_tool == "scale":
                self.annotation_window.tools[selected_tool].dialog.cleanup()
                self.annotation_window.set_selected_tool(None)
            
            # Deactivate patch sampling tool if active
            if selected_tool == "patch_sampling":
                self.annotation_window.tools[selected_tool].dialog.cleanup()
                self.annotation_window.set_selected_tool(None)
            
            # Deactivate rugosity tool if active
            if selected_tool == "rugosity":
                self.annotation_window.tools[selected_tool].dialog.cleanup()
                self.annotation_window.set_selected_tool(None)
    
    def handle_image_changed(self):
        """Handle actions needed when the image is changed."""
        if self.annotation_window.selected_tool == 'sam':
            self.annotation_window.tools['sam'].cancel_working_area()
        if self.annotation_window.selected_tool == 'see_anything':
            self.annotation_window.tools['see_anything'].cancel_working_area()
        
        # Update label tooltips with current counts
        self.label_window.update_tooltips()


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