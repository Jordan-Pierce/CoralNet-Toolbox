# coralnet_toolbox/Explorer/QtEmbeddingViewerWindow.py
"""
Standalone Embedding Viewer Window.

This module provides a fully self-contained embedding visualization window
that integrates directly with MainWindow as a dockable widget. It combines
the scatter plot visualization with built-in ML pipeline controls.
"""

import os
import warnings

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

try:
    # Sometimes hard to install on Mac
    from umap import UMAP
except ImportError:
    UMAP = None

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal, pyqtSlot, QSignalBlocker
from PyQt5.QtGui import QColor, QPen, QPainter, QBrush, QPainterPath, QMouseEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QComboBox,
    QLabel, QPushButton, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QSizePolicy, QMessageBox, QApplication
)

from coralnet_toolbox.Explorer.core.QtDataItem import EmbeddingPointItem
from coralnet_toolbox.Explorer.core.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.managers.QtCacheManager import CacheManager
from coralnet_toolbox.Explorer.models.yolo_models import YOLO_MODELS, is_yolo_model
from coralnet_toolbox.Explorer.models.transformer_models import TRANSFORMER_MODELS, is_transformer_model
from coralnet_toolbox.Explorer.workers import EmbeddingPipelineWorker

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox.utilities import pixmap_to_numpy, pixmap_to_pil

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

REVIEW_LABEL = 'Review'
POINT_WIDTH = 3


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingViewerWindow(QWidget):
    """
    Standalone embedding visualization window for ML-based annotation exploration.
    
    This widget is designed to be wrapped by DockWrapper and integrated into
    MainWindow as a persistent dock. It owns the CacheManager, handles feature
    extraction, dimensionality reduction, and visualization.
    
    Signals:
        selection_changed (list): Emitted when points are selected/deselected.
        embedding_complete (): Emitted when embedding pipeline finishes.
    """
    
    selection_changed = pyqtSignal(list)  # List of annotation IDs
    embedding_complete = pyqtSignal()
    reset_view_requested = pyqtSignal()
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the EmbeddingViewerWindow.
        
        Args:
            main_window: Reference to the MainWindow instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window
        
        # Animation manager reference
        self.animation_manager = None
        
        # Sync state
        self._syncing_selection = False  # Flag to prevent selection sync loops
        self._embeddings_stale = False  # Flag indicating new annotations need embedding
        
        # Cache manager for caching extracted features
        self.cache_manager = CacheManager()
        
        # Data model
        self.data_item_cache = {}  # annotation_id -> AnnotationDataItem
        self.working_set_ids = []  # List of annotation IDs to embed
        self.current_data_items = []  # Currently embedded data items
        self.current_features = None  # Features for current embedding
        self.current_feature_model_key = None  # Model key used for features
        
        # Model caching
        self._cached_yolo_model = None
        self._cached_yolo_model_name = None
        self._cached_transformer_model = None
        self._cached_transformer_model_name = None
        
        # Device and image size settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imgsz = 224
        
        # Points tracking
        self.points_by_id = {}
        self.previous_selection_ids = set()
        
        # State for pseudo-3D rotation
        self.is_rotating = False
        self.last_mouse_pos = QPointF()
        self.rotation_angle_y = 0.0
        self.rotation_angle_x = 0.0
        self.min_z = 0.0
        self.max_z = 0.0
        self.z_range = 0.0
        self.is_3d_data = False
        
        # Rubber band selection
        self.rubber_band = None
        self.rubber_band_origin = QPointF()
        self.selection_at_press = None
        
        # Isolation state
        self.isolated_mode = False
        self.isolated_points = set()
        
        # Selection blocking
        self.selection_blocked = False
        
        # Display mode
        self.display_mode = 'dots'  # 'dots' or 'sprites'
        
        # Location indicator
        self.locate_lines = []
        self.locate_graphics_item = None
        self.locate_timer = QTimer(self)
        self.locate_timer.setSingleShot(True)
        self.locate_timer.timeout.connect(self._clear_location_indicator)
        
        # Virtualization timer
        self.view_update_timer = QTimer(self)
        self.view_update_timer.setSingleShot(True)
        self.view_update_timer.timeout.connect(self._update_visible_points)
        
        # Background worker for embedding pipeline
        self._pipeline_worker = None
        self._pipeline_running = False
        
        # Build UI
        self._setup_ui()
        
    def set_animation_manager(self, manager):
        """Set the animation manager for visual effects."""
        self.animation_manager = manager
        
    # -------------------------------------------------------------------------
    # Toolbar Creation (for DockWrapper integration)
    # -------------------------------------------------------------------------
    
    def create_top_toolbar(self) -> QToolBar:
        """Create the top toolbar with analysis tools."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        
        # Isolate Selection button
        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setToolTip("Show only selected points (double-click to exit)")
        self.isolate_button.clicked.connect(self._isolate_selection)
        self.isolate_button.setEnabled(False)
        toolbar.addWidget(self.isolate_button)
        
        toolbar.addSeparator()
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        toolbar.addSeparator()
        
        # Locate button
        self.locate_button = QPushButton()
        self.locate_button.setIcon(get_icon("location.svg"))
        self.locate_button.setToolTip("Show location indicator for selected annotation")
        self.locate_button.clicked.connect(self._on_locate_clicked)
        toolbar.addWidget(self.locate_button)
        
        # Center on selection button
        self.center_button = QPushButton()
        self.center_button.setIcon(get_icon("target.svg"))
        self.center_button.setToolTip("Center view on selected point(s)")
        self.center_button.clicked.connect(self._center_on_selection)
        toolbar.addWidget(self.center_button)
        
        # Home button
        self.home_button = QPushButton()
        self.home_button.setIcon(get_icon("home.svg"))
        self.home_button.setToolTip("Reset view to fit all points")
        self.home_button.clicked.connect(self._reset_view)
        toolbar.addWidget(self.home_button)
        
        # Sprite toggle button
        self.sprite_toggle_button = QPushButton()
        self.sprite_toggle_button.setIcon(get_icon("sprites.svg"))
        self.sprite_toggle_button.setToolTip("Switch to Sprites View")
        self.sprite_toggle_button.clicked.connect(self._on_display_mode_changed)
        toolbar.addWidget(self.sprite_toggle_button)
        
        return toolbar
    
    def create_bottom_toolbar(self) -> QToolBar:
        """Create the bottom toolbar with ML pipeline controls."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        
        # Model Category
        category_label = QLabel(" Model: ")
        toolbar.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Color Features", "YOLO", "Transformer"])
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        toolbar.addWidget(self.category_combo)
        
        # Model Selection (dynamically populated)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        toolbar.addWidget(self.model_combo)
        
        toolbar.addSeparator()
        
        # Embedding Technique
        technique_label = QLabel(" Technique: ")
        toolbar.addWidget(technique_label)
        
        available_techniques = []
        if PCA:
            available_techniques.append("PCA")
        if LDA:
            available_techniques.append("LDA")
        if TSNE:
            available_techniques.append("TSNE")
        if UMAP:
            available_techniques.append("UMAP")
            
        self.technique_combo = QComboBox()
        self.technique_combo.addItems(available_techniques)
        toolbar.addWidget(self.technique_combo)
        
        # Dimensions
        dims_label = QLabel(" Dims: ")
        toolbar.addWidget(dims_label)
        
        self.dimensions_combo = QComboBox()
        self.dimensions_combo.addItems(["2D", "3D"])
        toolbar.addWidget(self.dimensions_combo)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Run button
        self.run_button = QPushButton("Run Embedding")
        self.run_button.setToolTip("Extract features and generate embedding visualization")
        self.run_button.clicked.connect(self.run_embedding_pipeline)
        toolbar.addWidget(self.run_button)
        
        # Initialize model combo based on default category
        self._on_category_changed(self.category_combo.currentText())
        
        return toolbar
    
    def _on_category_changed(self, category):
        """Update model combo based on selected category."""
        self.model_combo.clear()
        
        if category == "Color Features":
            self.model_combo.setEnabled(False)
            self.model_combo.addItem("N/A")
        elif category == "YOLO":
            self.model_combo.setEnabled(True)
            for name in YOLO_MODELS.keys():
                self.model_combo.addItem(name)
        elif category == "Transformer":
            self.model_combo.setEnabled(True)
            for name in TRANSFORMER_MODELS.keys():
                self.model_combo.addItem(name)
    
    # -------------------------------------------------------------------------
    # UI Setup
    # -------------------------------------------------------------------------
    
    def _setup_ui(self):
        """Setup the main UI layout with graphics view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Graphics scene and view
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)
        self.graphics_scene.selectionChanged.connect(self._on_selection_changed)
        
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMinimumHeight(200)
        
        # Override mouse events
        self.graphics_view.mousePressEvent = self._mouse_press_event
        self.graphics_view.mouseDoubleClickEvent = self._mouse_double_click_event
        self.graphics_view.mouseReleaseEvent = self._mouse_release_event
        self.graphics_view.mouseMoveEvent = self._mouse_move_event
        self.graphics_view.wheelEvent = self._wheel_event
        
        layout.addWidget(self.graphics_view)
        
        # Placeholder label
        self.placeholder_label = QLabel(
            "No embedding data available.\nSelect data from Annotation Gallery and click 'Run Embedding'."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(self.placeholder_label)
        
        self._show_placeholder()
        
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def set_working_set(self, annotation_ids):
        """
        Set the working set of annotations to embed.
        
        Called by MainWindow when the gallery filter changes.
        
        If the embedding viewer currently has points displayed, this will
        check if the new working set matches the current embedding. If not,
        it will clear the display and show the placeholder.
        
        Args:
            annotation_ids: List of annotation IDs to use for embedding.
        """
        self.working_set_ids = list(annotation_ids)
        
        # If we have points currently displayed, check if they match the new working set
        if self.points_by_id:
            working_set = set(annotation_ids)
            displayed_ids = set(self.points_by_id.keys())
            
            # If the working set doesn't exactly match what's displayed,
            # or if any annotation in working set is missing from display,
            # clear the embedding viewer
            if working_set != displayed_ids:
                self._clear_points()
                self._show_placeholder()
    
    def get_selected_annotation_ids(self):
        """Get list of currently selected annotation IDs."""
        return [p.data_item.annotation.id for p in self.graphics_scene.selectedItems()
                if isinstance(p, EmbeddingPointItem)]
    
    def highlight_points(self, ids):
        """Highlight specific points in the scatter plot."""
        self.render_selection_from_ids(set(ids))
        
    # -------------------------------------------------------------------------
    # Reactive Slots
    # -------------------------------------------------------------------------
    
    @pyqtSlot(str)
    def on_annotation_deleted(self, annotation_id):
        """Handle an annotation being deleted."""
        # Remove from cache
        if annotation_id in self.data_item_cache:
            del self.data_item_cache[annotation_id]
        
        # Remove from working set
        if annotation_id in self.working_set_ids:
            self.working_set_ids.remove(annotation_id)
        
        # Remove from current data items
        self.current_data_items = [
            item for item in self.current_data_items
            if item.annotation.id != annotation_id
        ]
        
        # Remove point from scene
        if annotation_id in self.points_by_id:
            point = self.points_by_id[annotation_id]
            self.graphics_scene.removeItem(point)
            del self.points_by_id[annotation_id]
        
        # Invalidate cached features
        self.cache_manager.remove_features_for_annotation(annotation_id)
        
        self._update_toolbar_state()
    
    @pyqtSlot(str, str)
    def on_annotation_label_changed(self, annotation_id, new_label):
        """Handle an annotation's label being changed."""
        if annotation_id in self.points_by_id:
            point = self.points_by_id[annotation_id]
            point.update()
    
    @pyqtSlot(str)
    def on_annotation_modified(self, annotation_id):
        """Handle annotation modification - invalidates cached features."""
        self.cache_manager.remove_features_for_annotation(annotation_id)
        
        if annotation_id in self.data_item_cache:
            del self.data_item_cache[annotation_id]
    
    @pyqtSlot(object)
    def on_annotation_selection_changed(self, selected_ids):
        """
        Handle selection changes from AnnotationWindow.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        if self._syncing_selection:
            return
        self._syncing_selection = True
        try:
            self.highlight_points(selected_ids if selected_ids else [])
        finally:
            self._syncing_selection = False
    
    @pyqtSlot(str)
    def on_annotation_created(self, annotation_id):
        """
        Handle new annotation created - flag that embeddings may need refresh.
        
        Args:
            annotation_id: ID of the newly created annotation.
        """
        self._embeddings_stale = True
    
    @pyqtSlot(object)
    def on_annotations_labels_changed(self, changes):
        """
        Handle batch label changes - just update visuals, don't invalidate features.
        
        Args:
            changes: List of tuples (annotation_id, old_label, new_label)
        """
        # Just update point visuals
        for annotation_id, old_label, new_label in changes:
            if annotation_id in self.points_by_id:
                point = self.points_by_id[annotation_id]
                point.update()
    
    @pyqtSlot(str, object)
    def on_annotation_moved(self, annotation_id, move_data):
        """
        Handle annotation being moved - might need to invalidate features if moved significantly.
        
        Args:
            annotation_id: ID of the moved annotation
            move_data: Dict with 'old_center' and 'new_center' QPointF
        """
        # For now, invalidate features on any move since the crop might change
        self.cache_manager.remove_features_for_annotation(annotation_id)
        if annotation_id in self.data_item_cache:
            del self.data_item_cache[annotation_id]
        self._embeddings_stale = True
    
    @pyqtSlot(str, object)
    def on_annotation_geometry_edited(self, annotation_id, geometry_data):
        """
        Handle annotation geometry being edited - invalidate cached features.
        
        Args:
            annotation_id: ID of the annotation
            geometry_data: Dict with 'old_geom' and 'new_geom'
        """
        self.cache_manager.remove_features_for_annotation(annotation_id)
        if annotation_id in self.data_item_cache:
            del self.data_item_cache[annotation_id]
        self._embeddings_stale = True
    
    @pyqtSlot(str, object)
    def on_annotation_cut(self, original_annotation_id, new_annotations):
        """
        Handle annotation being cut - remove original and mark stale.
        
        Args:
            original_annotation_id: ID of the original annotation
            new_annotations: List of new annotation objects
        """
        # Remove original from cache
        self.cache_manager.remove_features_for_annotation(original_annotation_id)
        if original_annotation_id in self.data_item_cache:
            del self.data_item_cache[original_annotation_id]
        
        # Remove from working set
        if original_annotation_id in self.working_set_ids:
            self.working_set_ids.remove(original_annotation_id)
        
        # Remove point from scene
        if original_annotation_id in self.points_by_id:
            point = self.points_by_id[original_annotation_id]
            self.graphics_scene.removeItem(point)
            del self.points_by_id[original_annotation_id]
        
        self._embeddings_stale = True
    
    @pyqtSlot(object)
    def on_annotations_merged(self, merge_data):
        """
        Handle multiple annotations being merged - remove originals and mark stale.
        
        Args:
            merge_data: Dict with 'original_ids' list and 'merged' annotation object
        """
        original_ids = merge_data['original_ids']
        
        # Remove originals from cache
        for ann_id in original_ids:
            self.cache_manager.remove_features_for_annotation(ann_id)
            if ann_id in self.data_item_cache:
                del self.data_item_cache[ann_id]
            
            # Remove from working set
            if ann_id in self.working_set_ids:
                self.working_set_ids.remove(ann_id)
            
            # Remove point from scene
            if ann_id in self.points_by_id:
                point = self.points_by_id[ann_id]
                self.graphics_scene.removeItem(point)
                del self.points_by_id[ann_id]
        
        self._embeddings_stale = True
    
    @pyqtSlot(str, object)
    def on_annotation_split(self, original_annotation_id, new_annotations):
        """
        Handle annotation being split - same as cut.
        
        Args:
            original_annotation_id: ID of the original annotation
            new_annotations: List of new annotation objects
        """
        self.on_annotation_cut(original_annotation_id, new_annotations)
    
    @pyqtSlot(str)
    def on_image_loaded(self, image_path):
        """
        Handle when a new image is loaded in ImageWindow.
        
        Clears the embedding viewer since the displayed embeddings are no longer
        relevant to the current image (unless filter is "All Images").
        
        Args:
            image_path: Path to the loaded image.
        """
        # Check if annotation viewer is filtering by specific image or showing all
        annotation_viewer = getattr(self.main_window, 'annotation_viewer_window', None)
        if annotation_viewer and hasattr(annotation_viewer, 'image_filter_combo'):
            current_filter = annotation_viewer.image_filter_combo.currentData()
            
            # If filter is "All Images", the embedding viewer can stay as-is
            # since it may contain annotations from multiple images
            if current_filter == "all":
                return
        
        # Clear the embedding viewer since we're filtering by specific image
        # and the current embeddings are from a different image
        self._clear_points()
        self._show_placeholder()
        
        # Reset state
        self.current_data_items = []
        self.current_features = None
        self._embeddings_stale = True
    
    # -------------------------------------------------------------------------
    # Embedding Pipeline
    # -------------------------------------------------------------------------
    
    def run_embedding_pipeline(self):
        """
        Execute the full embedding pipeline in a background thread:
        1. Get working set from gallery
        2. Extract features
        3. Run dimensionality reduction
        4. Update visualization
        """
        # Cancel any existing pipeline
        if self._pipeline_running and self._pipeline_worker:
            self._pipeline_worker.cancel()
            self._pipeline_worker.wait()
        
        # Get working set from gallery if available
        if hasattr(self.main_window, 'annotation_viewer_window'):
            self.working_set_ids = (
                self.main_window.annotation_viewer_window.get_currently_displayed_annotations()
            )
        
        if not self.working_set_ids:
            QMessageBox.information(
                self, "No Data",
                "No annotations to embed. Apply a filter in the Annotation Gallery first."
            )
            return
        
        # Get annotations from AnnotationWindow
        annotations = []
        for ann_id in self.working_set_ids:
            if hasattr(self.annotation_window, 'annotations_dict'):
                if ann_id in self.annotation_window.annotations_dict:
                    annotations.append(self.annotation_window.annotations_dict[ann_id])
        
        if not annotations:
            return
        
        # Ensure cropped images exist
        self._ensure_cropped_images(annotations)
        
        # Create/update data items
        data_items = []
        for ann in annotations:
            if ann.id not in self.data_item_cache:
                self.data_item_cache[ann.id] = AnnotationDataItem(ann)
            data_items.append(self.data_item_cache[ann.id])
        
        # Get model settings
        model_name = self._get_selected_model()
        embedding_params = self._get_embedding_parameters()
        
        # Generate model key for caching
        if os.path.sep in model_name or '/' in model_name:
            sanitized_model_name = os.path.basename(model_name)
        else:
            sanitized_model_name = model_name
        sanitized_model_name = sanitized_model_name.replace(' ', '_').replace('/', '_')
        model_key = sanitized_model_name
        
        # Create worker with callable extractors
        self._pipeline_worker = EmbeddingPipelineWorker(
            data_items=data_items,
            model_name=model_name,
            model_key=model_key,
            embedding_params=embedding_params,
            cache_manager=self.cache_manager,
            feature_extractor_fn=self._extract_features_for_worker,
            dim_reduction_fn=self._run_dimensionality_reduction
        )
        
        # Connect signals
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        
        # Start worker
        self._pipeline_running = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        if hasattr(self.main_window, 'statusBar') and callable(self.main_window.statusBar):
            self.main_window.statusBar().showMessage("Running embedding pipeline...")
        self._disable_analysis_buttons()
        self._pipeline_worker.start()
        
    def _extract_features_for_worker(self, model_name, data_items):
        """
        Wrapper for _extract_features that works with the worker thread.
        Note: This runs in the worker thread.
        
        Args:
            model_name: Name/path of the feature extraction model
            data_items: List of AnnotationDataItem objects
            
        Returns:
            tuple: (features_array, valid_items_list)
        """
        return self._extract_features(data_items, model_name, progress_bar=None)
    
    def _on_pipeline_progress(self, message):
        """Handle progress updates from the worker."""
        # Update main window status bar if available
        if hasattr(self.main_window, 'statusBar') and callable(self.main_window.statusBar):
            self.main_window.statusBar().showMessage(message)
    
    def _on_pipeline_finished(self, results):
        """Handle successful completion of the embedding pipeline."""
        try:
            # Extract results
            final_data_items = results['data_items']
            features = results['features']
            embedded_features = results['embedded_features']
            model_key = results['model_key']
            
            # Update state
            self.current_data_items = final_data_items
            self.current_features = features
            self.current_feature_model_key = model_key
            
            # Update visualization
            self._update_data_items_with_embedding(final_data_items, embedded_features)
            self._update_embeddings(final_data_items, embedded_features.shape[1])
            self._show_embedding()
            self._reset_view()
            
            self.embedding_complete.emit()
            
            if hasattr(self.main_window, 'statusBar') and callable(self.main_window.statusBar):
                self.main_window.statusBar().showMessage("Embedding complete", 3000)
                
        finally:
            self._pipeline_running = False
            QApplication.restoreOverrideCursor()
            self._update_toolbar_state()
    
    def _on_pipeline_error(self, error_message):
        """Handle errors from the worker."""
        QMessageBox.warning(self, "Pipeline Error", error_message)
        
        self._pipeline_running = False
        QApplication.restoreOverrideCursor()
        self._update_toolbar_state()
        
        if hasattr(self.main_window, 'statusBar') and callable(self.main_window.statusBar):
            self.main_window.statusBar().showMessage("Embedding failed", 3000)
    
    def _get_selected_model(self):
        """Get the currently selected model name/path."""
        category = self.category_combo.currentText()
        
        if category == "Color Features":
            return "Color Features"
        elif category == "YOLO":
            display_name = self.model_combo.currentText()
            return YOLO_MODELS.get(display_name, display_name)
        elif category == "Transformer":
            display_name = self.model_combo.currentText()
            return TRANSFORMER_MODELS.get(display_name, display_name)
        return ""
    
    def _get_embedding_parameters(self):
        """Get current embedding parameters from UI."""
        params = {
            'technique': self.technique_combo.currentText(),
            'dimensions': 3 if self.dimensions_combo.currentText() == "3D" else 2,
            'perform_pca_before': True,
            'pca_components': 50,
        }
        # Default parameters for techniques
        if params['technique'] == 'UMAP':
            params['n_neighbors'] = 15
            params['min_dist'] = 0.1
        elif params['technique'] == 'TSNE':
            params['perplexity'] = 30
            params['early_exaggeration'] = 12.0
        return params
    
    def _ensure_cropped_images(self, annotations):
        """Ensure cropped images are available for annotations."""
        for ann in annotations:
            if not hasattr(ann, 'cropped_image') or ann.cropped_image is None:
                try:
                    ann.create_cropped_image(self.annotation_window.image_pixmap)
                except Exception:
                    pass
    
    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------
    
    def _extract_features(self, data_items, model_name, progress_bar=None):
        """Dispatch to appropriate feature extraction method."""
        if model_name == "Color Features":
            return self._extract_color_features(data_items, progress_bar)
        elif is_yolo_model(model_name):
            return self._extract_yolo_features(data_items, model_name, progress_bar)
        elif is_transformer_model(model_name):
            return self._extract_transformer_features(data_items, model_name, progress_bar)
        return np.array([]), []
    
    def _extract_color_features(self, data_items, progress_bar=None, bins=32):
        """Extract color histogram features from annotation crops."""
        if progress_bar:
            progress_bar.set_title("Extracting color features...")
            progress_bar.start_progress(len(data_items))
        
        features = []
        valid_items = []
        
        for item in data_items:
            try:
                ann = item.annotation
                if not hasattr(ann, 'cropped_image') or ann.cropped_image is None:
                    if progress_bar:
                        progress_bar.update_progress()
                    continue
                
                img = pixmap_to_numpy(ann.cropped_image)
                if img is None or img.size == 0:
                    if progress_bar:
                        progress_bar.update_progress()
                    continue
                
                # Calculate histograms for each channel
                hist_features = []
                for i in range(min(3, img.shape[2]) if len(img.shape) > 2 else 1):
                    if len(img.shape) > 2:
                        channel = img[:, :, i]
                    else:
                        channel = img
                    hist, _ = np.histogram(channel.flatten(), bins=bins, range=(0, 256))
                    hist = hist.astype(np.float32)
                    hist /= (hist.sum() + 1e-7)
                    hist_features.extend(hist)
                
                features.append(hist_features)
                valid_items.append(item)
                
            except Exception:
                pass
            finally:
                if progress_bar:
                    progress_bar.update_progress()
        
        return np.array(features), valid_items
    
    def _extract_yolo_features(self, data_items, model_name, progress_bar=None):
        """Extract features using YOLO model."""
        model = self._load_yolo_model(model_name)
        if model is None:
            return np.array([]), []
        
        # Prepare images
        image_list, valid_items = self._prepare_images(data_items, progress_bar, 'numpy')
        if not valid_items:
            return np.array([]), []
        
        kwargs = {
            'stream': True,
            'imgsz': self.imgsz,
            'half': True,
            'device': self.device,
            'verbose': False
        }
        
        results_generator = model.embed(image_list, **kwargs)
        
        if progress_bar:
            progress_bar.set_title("Extracting features...")
            progress_bar.start_progress(len(valid_items))
        
        try:
            features_list = []
            for result in results_generator:
                if isinstance(result, list):
                    feature_vector = result[0].cpu().numpy().flatten()
                else:
                    feature_vector = result.cpu().numpy().flatten()
                features_list.append(feature_vector)
                if progress_bar:
                    progress_bar.update_progress()
            
            return np.array(features_list), valid_items
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Feature extraction failed: {e}")
            return np.array([]), []
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _extract_transformer_features(self, data_items, model_name, progress_bar=None):
        """Extract features using transformer model."""
        try:
            if progress_bar:
                progress_bar.set_busy_mode(f"Loading model {model_name}...")
            
            feature_extractor = self._load_transformer_model(model_name)
            if feature_extractor is None:
                return np.array([]), []
            
            image_list, valid_items = self._prepare_images(data_items, progress_bar, 'pil')
            if not image_list:
                return np.array([]), []
            
            if progress_bar:
                progress_bar.set_title("Extracting features...")
                progress_bar.start_progress(len(valid_items))
            
            features_list = []
            valid_results = []
            
            for i, image in enumerate(image_list):
                try:
                    features = feature_extractor(image)
                    feature_tensor = features[0] if isinstance(features, list) else features
                    
                    if hasattr(feature_tensor, 'cpu'):
                        feature_vector = feature_tensor.cpu().numpy().flatten()
                    else:
                        feature_vector = np.array(feature_tensor).flatten()
                    
                    features_list.append(feature_vector)
                    valid_results.append(valid_items[i])
                except Exception:
                    pass
                finally:
                    if progress_bar:
                        progress_bar.update_progress()
            
            return np.array(features_list), valid_results
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Transformer extraction failed: {e}")
            return np.array([]), []
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _prepare_images(self, data_items, progress_bar, format_type):
        """Prepare images from data items for model input."""
        if progress_bar:
            progress_bar.set_title("Preparing images...")
            progress_bar.start_progress(len(data_items))
        
        images = []
        valid_items = []
        
        for item in data_items:
            try:
                ann = item.annotation
                if not hasattr(ann, 'cropped_image') or ann.cropped_image is None:
                    continue
                
                if format_type == 'numpy':
                    img = pixmap_to_numpy(ann.cropped_image)
                    if img is not None:
                        from PIL import Image
                        pil_img = Image.fromarray(img)
                        pil_img = pil_img.resize((self.imgsz, self.imgsz))
                        images.append(np.array(pil_img))
                        valid_items.append(item)
                else:  # pil
                    img = pixmap_to_pil(ann.cropped_image)
                    if img is not None:
                        img = img.resize((self.imgsz, self.imgsz))
                        images.append(img)
                        valid_items.append(item)
            except Exception:
                pass
            finally:
                if progress_bar:
                    progress_bar.update_progress()
        
        return images, valid_items
    
    def _load_yolo_model(self, model_name):
        """Load YOLO model with caching."""
        if self._cached_yolo_model_name == model_name and self._cached_yolo_model:
            return self._cached_yolo_model
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            self._cached_yolo_model = model
            self._cached_yolo_model_name = model_name
            return model
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load YOLO model: {e}")
            return None
    
    def _load_transformer_model(self, model_name):
        """Load transformer model with caching."""
        if self._cached_transformer_model_name == model_name and self._cached_transformer_model:
            return self._cached_transformer_model
        
        try:
            from transformers import pipeline
            feature_extractor = pipeline(
                "image-feature-extraction",
                model=model_name,
                device=0 if self.device == 'cuda' else -1
            )
            self._cached_transformer_model = feature_extractor
            self._cached_transformer_model_name = model_name
            return feature_extractor
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load transformer: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Dimensionality Reduction
    # -------------------------------------------------------------------------
    
    def _run_dimensionality_reduction(self, features, params):
        """Run dimensionality reduction on features."""
        technique = params.get('technique', 'UMAP')
        n_components = params.get('dimensions', 2)
        pca_components = params.get('pca_components', 50)
        perform_pca = params.get('perform_pca_before', True)
        
        if len(features) <= n_components:
            return None
        
        try:
            features_scaled = StandardScaler().fit_transform(features)
            
            # PCA preprocessing
            if perform_pca and technique != "PCA" and features_scaled.shape[1] > pca_components:
                pca_components = min(pca_components, features_scaled.shape[0] - 1, features_scaled.shape[1])
                pca = PCA(n_components=pca_components, random_state=42)
                features_scaled = pca.fit_transform(features_scaled)
            
            # LDA special handling
            if technique == "LDA":
                labels = []
                labeled_indices = []
                for i, item in enumerate(self.current_data_items):
                    label_name = getattr(item.effective_label, 'short_label_code', REVIEW_LABEL)
                    if label_name != REVIEW_LABEL:
                        labels.append(label_name)
                        labeled_indices.append(i)
                
                if len(set(labels)) < 2:
                    QMessageBox.warning(self, "LDA Error", "LDA requires at least 2 labeled classes.")
                    return None
                
                labeled_features = features_scaled[labeled_indices]
                n_components_lda = min(n_components, len(set(labels)) - 1)
                reducer = LDA(n_components=n_components_lda)
                reducer.fit(labeled_features, labels)
                return reducer.transform(features_scaled)
            
            # Other techniques
            if technique == "UMAP":
                n_neighbors = min(params.get('n_neighbors', 15), len(features_scaled) - 1)
                reducer = UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=params.get('min_dist', 0.1),
                    metric='cosine'
                )
            elif technique == "TSNE":
                perplexity = min(params.get('perplexity', 30), len(features_scaled) - 1)
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=perplexity,
                    early_exaggeration=params.get('early_exaggeration', 12.0),
                    learning_rate='auto',
                    init='pca'
                )
            elif technique == "PCA":
                reducer = PCA(n_components=n_components, random_state=42)
            else:
                return None
            
            return reducer.fit_transform(features_scaled)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Dimensionality reduction failed: {e}")
            return None
    
    def _update_data_items_with_embedding(self, data_items, embedded_features):
        """Update data items with embedding coordinates."""
        if embedded_features is None:
            return
        
        n_dims = embedded_features.shape[1]
        scale_factor = 4000
        min_vals = np.min(embedded_features, axis=0)
        max_vals = np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        for i, item in enumerate(data_items):
            norm_coords = (embedded_features[i] - min_vals) / range_vals
            scaled_coords = (norm_coords * scale_factor) - (scale_factor / 2)
            
            if n_dims == 3:
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = scaled_coords[2]
            else:
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = 0.0
            
            item.embedding_x = item.embedding_x_3d
            item.embedding_y = item.embedding_y_3d
            item.embedding_z = item.embedding_z_3d
            item.embedding_id = i
    
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    
    def _update_embeddings(self, data_items, n_dims):
        """Update the embedding visualization."""
        self._clear_points()
        self.is_3d_data = (n_dims == 3)
        
        for item in data_items:
            point = EmbeddingPointItem(item, self)
            point.set_animation_manager(self.animation_manager)
            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point
        
        self._apply_rotation_and_projection()
        self._update_toolbar_state()
        self._update_visible_points()
    
    def _clear_points(self):
        """Clear all points from scene."""
        if self.isolated_mode:
            self._show_all_points()
        
        for point in self.points_by_id.values():
            self.graphics_scene.removeItem(point)
        self.points_by_id.clear()
        self._update_toolbar_state()
    
    def _show_placeholder(self):
        """Show placeholder, hide graphics view."""
        self.graphics_view.setVisible(False)
        self.placeholder_label.setVisible(True)
        self._disable_analysis_buttons()
    
    def _show_embedding(self):
        """Show graphics view, hide placeholder."""
        self.graphics_view.setVisible(True)
        self.placeholder_label.setVisible(False)
        self._update_toolbar_state()
    
    def _disable_analysis_buttons(self):
        """Disable analysis buttons when no data."""
        if hasattr(self, 'locate_button'):
            self.locate_button.setEnabled(False)
        if hasattr(self, 'center_button'):
            self.center_button.setEnabled(False)
        if hasattr(self, 'isolate_button'):
            self.isolate_button.setEnabled(False)
    
    def _update_toolbar_state(self):
        """Update toolbar button states based on current state."""
        # Check for isolate button
        if not hasattr(self, 'isolate_button'):
            return
        
        selection_exists = bool(self.graphics_scene.selectedItems())
        points_exist = bool(self.points_by_id)
        
        # Update analysis buttons if they exist
        if hasattr(self, 'locate_button'):
            self.locate_button.setEnabled(points_exist and selection_exists)
        if hasattr(self, 'center_button'):
            self.center_button.setEnabled(points_exist and selection_exists)
        
        # Isolate button: enabled only when NOT in isolation mode AND has selection
        # When isolated, button is disabled (user exits via double-click)
        self.isolate_button.setEnabled(not self.isolated_mode and selection_exists)
    
    def _schedule_view_update(self):
        """Schedule delayed view update for virtualization."""
        self.view_update_timer.start(50)
    
    def _update_visible_points(self):
        """Update visibility of points based on viewport."""
        if self.isolated_mode or not self.points_by_id:
            return
        
        visible_rect = self.graphics_view.mapToScene(
            self.graphics_view.viewport().rect()
        ).boundingRect()
        
        buffer_x = visible_rect.width() * 0.2
        buffer_y = visible_rect.height() * 0.2
        buffered_rect = visible_rect.adjusted(-buffer_x, -buffer_y, buffer_x, buffer_y)
        
        for point in self.points_by_id.values():
            point.setVisible(buffered_rect.contains(point.pos()) or point.isSelected())
    
    # -------------------------------------------------------------------------
    # Selection Management
    # -------------------------------------------------------------------------
    
    def render_selection_from_ids(self, selected_ids):
        """Update visual selection from ID set."""
        blocker = QSignalBlocker(self.graphics_scene)
        
        for ann_id, point in self.points_by_id.items():
            is_selected = ann_id in selected_ids
            point.data_item.set_selected(is_selected)
            point.setSelected(is_selected)
        
        blocker.unblock()
        self._on_selection_changed()
        self._update_visible_points()
    
    def _on_selection_changed(self):
        """Handle selection changes in scene."""
        if not self.graphics_scene:
            return
        
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            return
        
        current_ids = {item.data_item.annotation.id for item in selected_items
                       if isinstance(item, EmbeddingPointItem)}
        
        if current_ids != self.previous_selection_ids:
            for point_id, point in self.points_by_id.items():
                point.data_item.set_selected(point_id in current_ids)
            
            self.selection_changed.emit(list(current_ids))
            self.previous_selection_ids = current_ids
        
        self._update_toolbar_state()
        self._schedule_view_update()
    
    # -------------------------------------------------------------------------
    # Isolation
    # -------------------------------------------------------------------------
    
    def _isolate_selection(self):
        """Hide non-selected points."""
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items or self.isolated_mode:
            return
        
        self.isolated_points = set(selected_items)
        self.graphics_view.setUpdatesEnabled(False)
        try:
            for point in self.points_by_id.values():
                point.setVisible(point in self.isolated_points)
            self.isolated_mode = True
        finally:
            self.graphics_view.setUpdatesEnabled(True)
        
        self._update_toolbar_state()
    
    def _show_all_points(self):
        """Show all points, exit isolation mode."""
        if not self.isolated_mode:
            return
        
        self.isolated_mode = False
        self.isolated_points.clear()
        self.graphics_view.setUpdatesEnabled(False)
        try:
            self._update_visible_points()
        finally:
            self.graphics_view.setUpdatesEnabled(True)
        
        self._update_toolbar_state()
    
    # -------------------------------------------------------------------------
    # View Controls
    # -------------------------------------------------------------------------
    
    def _reset_view(self):
        """Reset view to fit all points."""
        self.rotation_angle_y = 0.0
        self.rotation_angle_x = 0.0
        self._apply_rotation_and_projection()
        self._fit_view_to_points()
    
    def _fit_view_to_points(self):
        """Fit view to all points."""
        if self.points_by_id:
            self.graphics_view.fitInView(
                self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio
            )
        else:
            self.graphics_view.fitInView(-2500, -2500, 5000, 5000, Qt.KeepAspectRatio)
    
    def _center_on_selection(self):
        """Center view on selected points."""
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items:
            return
        
        selection_rect = None
        for item in selected_items:
            if isinstance(item, EmbeddingPointItem):
                item_rect = item.sceneBoundingRect()
                item_rect = item_rect.adjusted(-50, -50, 50, 50)
                if selection_rect is None:
                    selection_rect = item_rect
                else:
                    selection_rect = selection_rect.united(item_rect)
        
        if selection_rect:
            selection_rect = selection_rect.adjusted(-20, -20, 20, 20)
            self.graphics_view.fitInView(selection_rect, Qt.KeepAspectRatio)
            
            if self.locate_graphics_item:
                self._update_location_lines()
    
    def _on_locate_clicked(self):
        """Handle locate button click."""
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items:
            return
        
        first_item = selected_items[0]
        if isinstance(first_item, EmbeddingPointItem):
            self._show_annotation_location(first_item)
    
    def _show_annotation_location(self, graphics_item):
        """Show convergent lines to annotation location."""
        self._clear_location_indicator()
        self.locate_graphics_item = graphics_item
        QTimer.singleShot(50, self._update_location_lines)
        self.locate_timer.start(1500)
    
    def _update_location_lines(self):
        """Update location indicator lines."""
        from PyQt5.QtWidgets import QGraphicsLineItem
        from PyQt5.QtCore import QLineF
        
        if not self.locate_graphics_item:
            return
        
        for line in self.locate_lines:
            self.graphics_scene.removeItem(line)
        self.locate_lines.clear()
        
        target_pos = self.locate_graphics_item.pos()
        target_x, target_y = target_pos.x(), target_pos.y()
        
        visible_rect = self.graphics_view.mapToScene(
            self.graphics_view.viewport().rect()
        ).boundingRect()
        
        lines_data = [
            QLineF(target_x, visible_rect.top(), target_x, target_y),
            QLineF(target_x, visible_rect.bottom(), target_x, target_y),
            QLineF(visible_rect.left(), target_y, target_x, target_y),
            QLineF(visible_rect.right(), target_y, target_x, target_y),
        ]
        
        pen = QPen(QColor(0, 0, 0), 3, Qt.DashLine)
        pen.setCosmetic(True)
        
        for line_data in lines_data:
            line_item = QGraphicsLineItem(line_data)
            line_item.setPen(pen)
            self.graphics_scene.addItem(line_item)
            self.locate_lines.append(line_item)
    
    def _clear_location_indicator(self):
        """Clear location indicator lines."""
        for line in self.locate_lines:
            self.graphics_scene.removeItem(line)
        self.locate_lines.clear()
        self.locate_graphics_item = None
        self.locate_timer.stop()
    
    def _on_display_mode_changed(self):
        """Toggle between dots and sprites view."""
        if self.display_mode == 'dots':
            self.display_mode = 'sprites'
            self.sprite_toggle_button.setIcon(get_icon("dot.svg"))
            self.sprite_toggle_button.setToolTip("Switch to Dots View")
        else:
            self.display_mode = 'dots'
            self.sprite_toggle_button.setIcon(get_icon("sprites.svg"))
            self.sprite_toggle_button.setToolTip("Switch to Sprites View")
        
        for point in self.points_by_id.values():
            point.prepareGeometryChange()
        self.graphics_scene.update()
    
    # -------------------------------------------------------------------------
    # 3D Rotation
    # -------------------------------------------------------------------------
    
    def _apply_rotation_and_projection(self):
        """Apply rotation to 3D points."""
        if not self.points_by_id:
            return
        
        point_items = list(self.points_by_id.values())
        original_points_3d = np.array([
            [p.data_item.embedding_x_3d, p.data_item.embedding_y_3d, p.data_item.embedding_z_3d]
            for p in point_items
        ])
        
        theta_x = np.radians(self.rotation_angle_x)
        theta_y = np.radians(self.rotation_angle_y)
        
        cos_x, sin_x = np.cos(theta_x), np.sin(theta_x)
        cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
        
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        
        rotation_matrix = rot_x @ rot_y
        rotated_points = original_points_3d @ rotation_matrix.T
        
        if len(rotated_points) > 0:
            self.min_z = np.min(rotated_points[:, 2])
            self.max_z = np.max(rotated_points[:, 2])
            self.z_range = self.max_z - self.min_z
        
        self.graphics_view.setUpdatesEnabled(False)
        try:
            for i, point in enumerate(point_items):
                rotated = rotated_points[i]
                point.data_item.embedding_x = rotated[0]
                point.data_item.embedding_y = rotated[1]
                point.data_item.embedding_z = rotated[2]
                point.prepareGeometryChange()
                point.setPos(rotated[0], rotated[1])
        finally:
            self.graphics_view.setUpdatesEnabled(True)
        
        self.graphics_scene.update()
    
    # -------------------------------------------------------------------------
    # Mouse Event Handlers
    # -------------------------------------------------------------------------
    
    def _mouse_press_event(self, event):
        """Handle mouse press events."""
        if self.selection_blocked:
            if event.button() == Qt.RightButton and event.modifiers() != Qt.ControlModifier:
                self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
                left_event = QMouseEvent(
                    event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()
                )
                QGraphicsView.mousePressEvent(self.graphics_view, left_event)
                return
            event.ignore()
            return
        
        # Ctrl+Right-Click for rotation (on empty space) or context menu (on point)
        if event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            
            # Ctrl+Right-Click on a point: navigate to annotation in AnnotationWindow
            if isinstance(item_at_pos, EmbeddingPointItem):
                self.graphics_scene.clearSelection()
                item_at_pos.setSelected(True)
                self._on_selection_changed()
                
                # Use SelectionManager if available for context menu navigation
                ann_id = item_at_pos.data_item.annotation.id
                if hasattr(self.main_window, 'selection_manager'):
                    self.main_window.selection_manager.handle_context_menu_selection(
                        ann_id, navigate_to=True
                    )
                else:
                    # Fallback: manually navigate to the annotation
                    annotation = item_at_pos.data_item.annotation
                    if hasattr(self, 'annotation_window') and self.annotation_window:
                        if self.annotation_window.current_image_path != annotation.image_path:
                            if hasattr(self.annotation_window, 'set_image'):
                                self.annotation_window.set_image(annotation.image_path)
                        if hasattr(self.annotation_window, 'select_annotation'):
                            self.annotation_window.select_annotation(annotation)
                        if hasattr(self.annotation_window, 'center_on_annotation'):
                            self.annotation_window.center_on_annotation(annotation)
                
                event.accept()
                return
            
            # Ctrl+Right-Click on empty space with 3D data: rotate
            if self.is_3d_data:
                self.is_rotating = True
                self.last_mouse_pos = event.pos()
                self.graphics_view.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        
        # Ctrl+Left-Click for rubber band selection
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                is_selected = item_at_pos.data_item.is_selected
                item_at_pos.data_item.set_selected(not is_selected)
                item_at_pos.setSelected(not is_selected)
                self._on_selection_changed()
                return
            
            self.selection_at_press = set(self.graphics_scene.selectedItems())
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.rubber_band_origin = self.graphics_view.mapToScene(event.pos())
            self.rubber_band = QGraphicsRectItem(
                QRectF(self.rubber_band_origin, self.rubber_band_origin)
            )
            pen = QPen(QColor(0, 168, 230), 3, Qt.DashLine)
            pen.setCosmetic(True)
            self.rubber_band.setPen(pen)
            self.rubber_band.setBrush(QBrush(QColor(0, 168, 230, 30)))
            self.graphics_scene.addItem(self.rubber_band)
        
        # Right-click for panning
        elif event.button() == Qt.RightButton:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = QMouseEvent(
                event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()
            )
            QGraphicsView.mousePressEvent(self.graphics_view, left_event)
        
        # Left-click (no modifiers) - clear selection when clicking empty space,
        # but DO NOT reset viewers (stay in isolated subset)
        elif event.button() == Qt.LeftButton and not event.modifiers():
            item_at_pos = self.graphics_view.itemAt(event.pos())
            
            # If clicked on a point, let default handling toggle selection
            if isinstance(item_at_pos, EmbeddingPointItem):
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                QGraphicsView.mousePressEvent(self.graphics_view, event)
            else:
                # Clicked on empty space - just clear selection without resetting viewers
                if self.graphics_scene.selectedItems():
                    self.graphics_scene.clearSelection()
                    self._on_selection_changed()
                event.accept()
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            QGraphicsView.mousePressEvent(self.graphics_view, event)
    
    def _mouse_double_click_event(self, event):
        """Handle double-click on empty space to clear selection and reset viewers (Show All)."""
        if self.selection_blocked:
            event.ignore()
            return
        
        if event.button() == Qt.LeftButton:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            
            # Only reset viewers if double-clicking on empty space
            if not isinstance(item_at_pos, EmbeddingPointItem):
                if self.graphics_scene.selectedItems():
                    self.graphics_scene.clearSelection()
                # Emit reset signal to exit isolation mode and show all in both viewers
                self.reset_view_requested.emit()
                event.accept()
            else:
                # Double-click on a point - could be used for other functionality
                # For now, just accept the event
                event.accept()
        else:
            QGraphicsView.mouseDoubleClickEvent(self.graphics_view, event)
    
    def _mouse_move_event(self, event):
        """Handle mouse move for rotation and rubber band."""
        if self.is_rotating:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.rotation_angle_y += delta.x() * 0.5
            self.rotation_angle_x += delta.y() * 0.5
            self._apply_rotation_and_projection()
            event.accept()
            return
        
        if self.rubber_band:
            current_pos = self.graphics_view.mapToScene(event.pos())
            self.rubber_band.setRect(
                QRectF(self.rubber_band_origin, current_pos).normalized()
            )
            path = QPainterPath()
            path.addRect(self.rubber_band.rect())
            self.graphics_scene.blockSignals(True)
            self.graphics_scene.setSelectionArea(path)
            if self.selection_at_press:
                for item in self.selection_at_press:
                    item.setSelected(True)
            self.graphics_scene.blockSignals(False)
            self._on_selection_changed()
        elif event.buttons() == Qt.RightButton:
            left_event = QMouseEvent(
                event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()
            )
            QGraphicsView.mouseMoveEvent(self.graphics_view, left_event)
            if self.locate_graphics_item:
                self._update_location_lines()
            self._schedule_view_update()
        else:
            QGraphicsView.mouseMoveEvent(self.graphics_view, event)
    
    def _mouse_release_event(self, event):
        """Handle mouse release."""
        if self.is_rotating:
            self.is_rotating = False
            self.graphics_view.unsetCursor()
            event.accept()
            return
        
        if self.selection_blocked:
            if self.rubber_band:
                self.graphics_scene.removeItem(self.rubber_band)
                self.rubber_band = None
                self.selection_at_press = None
            return
        
        if self.rubber_band:
            self.graphics_scene.removeItem(self.rubber_band)
            self.rubber_band = None
            self.selection_at_press = None
        elif event.button() == Qt.RightButton:
            left_event = QMouseEvent(
                event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers()
            )
            QGraphicsView.mouseReleaseEvent(self.graphics_view, left_event)
            self._schedule_view_update()
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        else:
            QGraphicsView.mouseReleaseEvent(self.graphics_view, event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
    
    def _wheel_event(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)
        
        old_pos = self.graphics_view.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.graphics_view.scale(zoom_factor, zoom_factor)
        new_pos = self.graphics_view.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.graphics_view.translate(delta.x(), delta.y())
        
        if self.locate_graphics_item:
            self._update_location_lines()
        
        self._schedule_view_update()
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def closeEvent(self, event):
        """Handle close event - cleanup resources."""
        self.cache_manager.close()
        super().closeEvent(event)
