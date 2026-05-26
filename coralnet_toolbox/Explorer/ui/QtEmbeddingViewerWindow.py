# coralnet_toolbox/Explorer/QtEmbeddingViewerWindow.py
"""
Standalone Embedding Viewer Window.

This module provides a fully self-contained embedding visualization window
that integrates directly with MainWindow as a dockable widget. It combines
the scatter plot visualization with built-in ML pipeline controls.
"""

import hashlib
from functools import partial
import os
import warnings
import time

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
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QToolBar, QComboBox,
    QLabel, QPushButton, QSpinBox, QSlider, QStackedWidget, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QSizePolicy, QMessageBox, QApplication
)

from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.Common.QtCollapsibleSection import CollapsibleSection

from coralnet_toolbox.Explorer.core.QtDataItem import EmbeddingPointItem, POINT_SIZE, SPRITE_SIZE
from coralnet_toolbox.Explorer.core.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.managers.CacheManager import CacheManager
from coralnet_toolbox.Explorer.models.ModelRegistry import YOLO_MODELS
from coralnet_toolbox.Explorer.models.ModelRegistry import is_live_yolo_model
from coralnet_toolbox.Explorer.models.ModelRegistry import is_yolo_model
from coralnet_toolbox.Explorer.models.ModelRegistry import TRANSFORMER_MODELS, is_transformer_model
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
        # Prefer a central cache_manager owned by MainWindow when available
        self.cache_manager = getattr(self.main_window, 'cache_manager', None)
        if self.cache_manager is None:
            # backward-compatible fallback
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

        # Advanced embedding settings UI state
        self.embedding_settings_section = None
        self.perform_pca_before_combo = None
        self.pca_components_spin = None
        self.technique_settings_stack = None
        self.umap_n_neighbors_row = None
        self.umap_n_neighbors_slider = None
        self.umap_min_dist_row = None
        self.umap_min_dist_slider = None
        self.tsne_perplexity_row = None
        self.tsne_perplexity_slider = None
        self.tsne_exaggeration_row = None
        self.tsne_exaggeration_slider = None
        
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
        # Dynamic sizing for points and sprites (Ctrl+Wheel will modify these)
        self.point_size = getattr(self, 'point_size', POINT_SIZE)
        self.sprite_size = getattr(self, 'sprite_size', SPRITE_SIZE)
        self._point_min = 4
        self._point_max = 128
        self._sprite_min = 16
        self._sprite_max = 512
        self._resize_step_point = 2
        self._resize_step_sprite = 8
        
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
        self.category_combo.addItems(["Color Features", "YOLO", "Transformer", "Live Models"])
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        toolbar.addWidget(self.category_combo)
        
        # Model Selection (dynamically populated)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(100)
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
        self.technique_combo.currentTextChanged.connect(self._on_technique_changed)
        toolbar.addWidget(self.technique_combo)
        
        # Dimensions
        dims_label = QLabel(" Dims: ")
        toolbar.addWidget(dims_label)
        
        self.dimensions_combo = QComboBox()
        self.dimensions_combo.addItems(["2D", "3D"])
        toolbar.addWidget(self.dimensions_combo)

        # Advanced settings stay hidden behind a popup so the toolbar remains compact.
        self.embedding_settings_section = CollapsibleSection(
            "Advanced",
            "parameters.svg",
            position='topright'
        )
        self._setup_embedding_settings_section()
        toolbar.addWidget(self.embedding_settings_section)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Clear button (next to Run Embedding) - clears only the embedding view
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear embedding view and reset placeholder")
        self.clear_button.clicked.connect(self.clear_view)
        toolbar.addWidget(self.clear_button)
        
        # Run button
        self.run_button = QPushButton("Run Embedding")
        self.run_button.setToolTip("Extract features and generate embedding visualization")
        self.run_button.clicked.connect(self.run_embedding_pipeline)
        toolbar.addWidget(self.run_button)

        # Initialize model combo based on default category
        self._on_category_changed(self.category_combo.currentText())
        self._on_technique_changed(self.technique_combo.currentText())
        
        return toolbar

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_model_options()

    def refresh_model_options(self):
        """Refresh the model combo while preserving the current selection when possible."""
        current_category = self.category_combo.currentText()
        selected_data = self.model_combo.currentData() if self.model_combo.count() else None
        self._populate_model_combo(current_category, selected_data)

    def _populate_model_combo(self, category, selected_data=None):
        """Populate the model combo for the active category."""
        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()

            if category == "Color Features":
                self.model_combo.setEnabled(False)
                self.model_combo.addItem("N/A", "Color Features")
            elif category == "YOLO":
                self.model_combo.setEnabled(True)
                for display_name, model_name in YOLO_MODELS.items():
                    self.model_combo.addItem(display_name, model_name)
            elif category == "Transformer":
                self.model_combo.setEnabled(True)
                for display_name, model_name in TRANSFORMER_MODELS.items():
                    self.model_combo.addItem(display_name, model_name)
            elif category == "Live Models":
                live_models = self._get_loaded_yolo_models()
                if live_models:
                    self.model_combo.setEnabled(True)
                    for source in live_models:
                        self.model_combo.addItem(
                            source.get("display_name", "Live Model"),
                            source.get("source_key", "")
                        )
                else:
                    self.model_combo.setEnabled(False)
                    self.model_combo.addItem("None loaded", "")
            else:
                self.model_combo.setEnabled(False)
                self.model_combo.addItem("N/A", "")

            if selected_data is not None:
                index = self.model_combo.findData(selected_data)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
        finally:
            self.model_combo.blockSignals(False)
    
    def _on_category_changed(self, category):
        """Update model combo based on selected category."""
        self._populate_model_combo(category)

    def _create_slider_row(self, minimum, maximum, value, formatter, tooltip, value_width=42):
        """Create a slider row with a live value label."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setToolTip(tooltip)

        value_label = QLabel(formatter(value))
        value_label.setMinimumWidth(value_width)

        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)

        slider.valueChanged.connect(
            lambda current_value, label=value_label, value_formatter=formatter: label.setText(value_formatter(current_value))
        )

        return row_widget, slider, value_label

    def _setup_embedding_settings_section(self):
        """Build the compact advanced settings popup for embedding controls."""
        advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(advanced_widget)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(8)

        general_group = QWidget()
        general_form = QFormLayout(general_group)
        general_form.setContentsMargins(0, 0, 0, 0)
        general_form.setSpacing(6)

        self.perform_pca_before_combo = QComboBox()
        self.perform_pca_before_combo.addItems(["True", "False"])
        self.perform_pca_before_combo.setCurrentText("True")
        self.perform_pca_before_combo.setToolTip(
            "Apply PCA before the final reduction step.\n"
            "Disabled for PCA because PCA is already the final reducer."
        )
        general_form.addRow("PCA before:", self.perform_pca_before_combo)

        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(1, 1000)
        self.pca_components_spin.setValue(50)
        self.pca_components_spin.setToolTip(
            "Number of PCA components used before TSNE, UMAP, or LDA.\n"
            "Lower values can speed up reduction on high-dimensional features."
        )
        general_form.addRow("PCA components:", self.pca_components_spin)

        advanced_layout.addWidget(general_group)

        self.technique_settings_stack = QStackedWidget()

        default_page = QWidget()
        default_layout = QVBoxLayout(default_page)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(0)
        default_label = QLabel("No additional settings for this technique.")
        default_label.setWordWrap(True)
        default_label.setStyleSheet("color: gray;")
        default_layout.addWidget(default_label)
        self.technique_settings_stack.addWidget(default_page)

        umap_page = QWidget()
        umap_form = QFormLayout(umap_page)
        umap_form.setContentsMargins(0, 0, 0, 0)
        umap_form.setSpacing(6)

        umap_n_neighbors_tooltip = (
            "Number of neighbors used by UMAP.\n"
            "Higher values capture more global structure; lower values focus on local structure."
        )
        self.umap_n_neighbors_row, self.umap_n_neighbors_slider, _ = self._create_slider_row(
            2, 150, 15,
            lambda current_value: str(current_value),
            umap_n_neighbors_tooltip,
            value_width=36
        )
        umap_form.addRow("n_neighbors:", self.umap_n_neighbors_row)

        umap_min_dist_tooltip = (
            "Minimum distance between points in the UMAP embedding.\n"
            "Smaller values pack points more tightly."
        )
        self.umap_min_dist_row, self.umap_min_dist_slider, _ = self._create_slider_row(
            0, 99, 10,
            lambda current_value: f"{current_value / 100.0:.2f}",
            umap_min_dist_tooltip,
            value_width=42
        )
        umap_form.addRow("min_dist:", self.umap_min_dist_row)

        self.technique_settings_stack.addWidget(umap_page)

        tsne_page = QWidget()
        tsne_form = QFormLayout(tsne_page)
        tsne_form.setContentsMargins(0, 0, 0, 0)
        tsne_form.setSpacing(6)

        tsne_perplexity_tooltip = (
            "Effective number of neighbors considered by t-SNE.\n"
            "Typical values range from 5 to 50."
        )
        self.tsne_perplexity_row, self.tsne_perplexity_slider, _ = self._create_slider_row(
            5, 50, 30,
            lambda current_value: str(current_value),
            tsne_perplexity_tooltip,
            value_width=36
        )
        tsne_form.addRow("Perplexity:", self.tsne_perplexity_row)

        tsne_exaggeration_tooltip = (
            "Controls how tightly clusters are separated in t-SNE.\n"
            "Larger values make clusters more distinct."
        )
        self.tsne_exaggeration_row, self.tsne_exaggeration_slider, _ = self._create_slider_row(
            50, 600, 120,
            lambda current_value: f"{current_value / 10.0:.1f}",
            tsne_exaggeration_tooltip,
            value_width=42
        )
        tsne_form.addRow("Exaggeration:", self.tsne_exaggeration_row)

        self.technique_settings_stack.addWidget(tsne_page)

        advanced_layout.addWidget(self.technique_settings_stack)
        advanced_layout.addStretch(1)

        self.embedding_settings_section.add_widget(advanced_widget, "Embedding Options")

    def _on_technique_changed(self, technique):
        """Update advanced controls when the reduction technique changes."""
        if self.perform_pca_before_combo is None:
            return

        use_general_pca_controls = technique != "PCA"
        self.perform_pca_before_combo.setEnabled(use_general_pca_controls)
        self.pca_components_spin.setEnabled(use_general_pca_controls)

        if technique == "UMAP":
            self.technique_settings_stack.setCurrentIndex(1)
        elif technique == "TSNE":
            self.technique_settings_stack.setCurrentIndex(2)
        else:
            self.technique_settings_stack.setCurrentIndex(0)

        if hasattr(self.embedding_settings_section, 'popup') and self.embedding_settings_section.popup.isVisible():
            self.embedding_settings_section.popup.adjustSize()
    
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
        # Override key press events
        self.graphics_view.keyPressEvent = self._key_press_event
        self.graphics_view.setStyleSheet(f"background-color: {app_theme.BACKGROUND_COLOR.name()};")
        self.graphics_scene.setBackgroundBrush(QColor(app_theme.BACKGROUND_COLOR))
        
        layout.addWidget(self.graphics_view)
        
        # Placeholder label
        self.placeholder_label = QLabel(
            "No embedding data available\nRun embedding to see visualizations."
        )
        self.placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setAutoFillBackground(True)
        self._show_placeholder()

        layout.addWidget(self.placeholder_label)

    def refresh_scaling(self):
        """Refresh the placeholder styling after a UI scale change."""
        self.graphics_view.setStyleSheet(f"background-color: {app_theme.BACKGROUND_COLOR.name()};")
        self.graphics_scene.setBackgroundBrush(QColor(app_theme.BACKGROUND_COLOR))
        self.placeholder_label.setStyleSheet(
            app_theme.scale_qss(
                f"color: {app_theme.TEXT_PRIMARY_COLOR.name()}; background-color: transparent; font-size: 14px; padding: 16px;"
            )
        )
        
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
        
        # Invalidate cached features (only when viewer is visible to avoid
        # churn when embedding/annotation views are inactive)
        if self.cache_manager and self._should_modify_cache():
            self.cache_manager.remove_features_for_annotation(annotation_id)
        
        self._update_toolbar_state()
        
    @pyqtSlot(list)
    def on_annotations_deleted(self, annotation_ids):
        """
        Handle a bulk deletion of annotations with optimized batch removal.
        """
        if not annotation_ids:
            return

        # 1. Use a set for O(1) lookups during the list rebuild
        ids_to_delete = set(annotation_ids)

        # 2. Block signals and updates to prevent individual redraws per item
        self.graphics_view.setUpdatesEnabled(False)
        self.graphics_scene.blockSignals(True)

        try:
            # 3. Fast Rebuild of the data item list
            self.current_data_items = [
                item for item in self.current_data_items
                if item.annotation.id not in ids_to_delete
            ]

            # 4. Remove Points and clean caches
            for ann_id in annotation_ids:
                # Clear visual points instantly
                if ann_id in self.points_by_id:
                    point = self.points_by_id.pop(ann_id)
                    self.graphics_scene.removeItem(point)
                
                # Cleanup internal caches
                self.data_item_cache.pop(ann_id, None)
                if ann_id in self.working_set_ids:
                    self.working_set_ids.remove(ann_id)

            # Invalidate cached ML features for deleted items in one transaction.
            if self.cache_manager and self._should_modify_cache():
                self.cache_manager.remove_features_for_annotations(annotation_ids)

        finally:
            # 5. Re-enable updates and perform ONE consolidated refresh
            self.graphics_scene.blockSignals(False)
            self.graphics_view.setUpdatesEnabled(True)
            self.graphics_scene.update()

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
        if self.cache_manager and self._should_modify_cache():
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

    @pyqtSlot(list)
    def on_annotations_created(self, annotation_ids):
        """
        Handle bulk annotation creation - flag that embeddings need refresh.
        """
        if annotation_ids:
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
        if self.cache_manager and self._should_modify_cache():
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
        if self.cache_manager and self._should_modify_cache():
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
        # Remove original from cache (guarded by visibility)
        if self.cache_manager and self._should_modify_cache():
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
        
        # Remove originals from cache (guarded by visibility)
        if self.cache_manager and self._should_modify_cache():
            self.cache_manager.remove_features_for_annotations(original_ids)

        for ann_id in original_ids:
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
        self.refresh_model_options()
        model_name = self._get_selected_model()
        if not model_name:
            QMessageBox.information(
                self,
                "No Model Selected",
                "No model is available for the selected category.",
            )
            return

        live_model = self._resolve_live_yolo_model(model_name)
        embedding_params = self._get_embedding_parameters()
        
        # Block LDA if only one class/label is selected in the annotation viewer
        try:
            if embedding_params.get('technique') == 'LDA':
                annotation_viewer = getattr(self.main_window, 'annotation_viewer_window', None)
                selected_labels = None
                if annotation_viewer is not None:
                    # Prefer the viewer's filter selection if available
                    try:
                        selected_labels = annotation_viewer._get_selected_labels()
                    except Exception:
                        selected_labels = None

                # If label filter is not restrictive, infer labels from the annotations
                labels_in_data = set()
                for ann in annotations:
                    try:
                        lbl = getattr(ann, 'label', None)
                        if lbl is None:
                            continue
                        code = getattr(lbl, 'short_label_code', None) or getattr(lbl, 'code', None) or str(lbl)
                        labels_in_data.add(code)
                    except Exception:
                        continue

                # Decide how many unique labels are effectively selected
                if selected_labels is None:
                    n_labels = len(labels_in_data)
                else:
                    # selected_labels may be a list of label codes
                    n_labels = len(selected_labels)

                if n_labels < 2:
                    QMessageBox.warning(
                        self,
                        "LDA Not Available",
                        "LDA requires at least two distinct classes. Select multiple labels in the Annotation Gallery or include multiple classes in the working set."
                    )
                    return
        except Exception:
            # If anything goes wrong during this check, fail-safe: allow pipeline to continue
            pass

        # Generate model key for caching
        model_key = self._get_model_cache_key(model_name)
        
        # Create worker with callable extractors
        self._pipeline_worker = EmbeddingPipelineWorker(
            data_items=data_items,
            model_name=model_name,
            model_key=model_key,
            embedding_params=embedding_params,
            cache_manager=self.cache_manager,
            feature_extractor_fn=partial(self._extract_features_for_worker, live_model=live_model),
            dim_reduction_fn=self._run_dimensionality_reduction
        )
        
        # Connect signals
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        
        # Start worker
        self._pipeline_running = True
        
        # Update status bar if available
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.main_window.status_bar.showMessage("Running embedding pipeline...")
        
        self._disable_analysis_buttons()
        self._pipeline_worker.start()
        
    def _extract_features_for_worker(self, model_name, data_items, live_model=None):
        """
        Wrapper for _extract_features that works with the worker thread.
        Note: This runs in the worker thread.
        
        Args:
            model_name: Name/path of the feature extraction model
            data_items: List of AnnotationDataItem objects
            
        Returns:
            tuple: (features_array, valid_items_list)
        """
        return self._extract_features(data_items, model_name, progress_bar=None, live_model=live_model)
    
    def _on_pipeline_progress(self, message):
        """Handle progress updates from the worker."""
        # Update main window status bar if available
        self.main_window.status_bar.showMessage(message)
    
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

            # Normalize embedded_features to an ndarray and compute dims safely
            embedded_features = np.asarray(embedded_features)
            if embedded_features.ndim == 1:
                # Convert shape (N,) to (N,1) if possible, or treat as single-dim
                embedded_features = embedded_features.reshape(-1, 1)

            n_dims = 1 if embedded_features.ndim == 1 else embedded_features.shape[1]

            # Update visualization
            self._update_data_items_with_embedding(final_data_items, embedded_features)
            self._update_embeddings(final_data_items, n_dims)
            self._show_embedding()
            self._reset_view()
            
            self.embedding_complete.emit()
            
            # Update status bar
            self.main_window.status_bar.showMessage("Embedding complete", 3000)
                
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
        
        # Update status bar with error message if available
        self.main_window.status_bar.showMessage("Embedding failed", 3000)
    
    def _get_selected_model(self):
        """Get the currently selected model name/path."""
        current_data = self.model_combo.currentData()
        if current_data not in (None, ""):
            return current_data
        return ""

    def _decode_live_yolo_source(self, model_name):
        """Decode a live YOLO source string into its parts."""
        if not is_live_yolo_model(model_name):
            return None

        parts = model_name.split("::", 3)
        if len(parts) != 4:
            return None

        _, dialog_key, task, model_path = parts
        if not dialog_key or not task or not model_path:
            return None

        return {
            "dialog_key": dialog_key,
            "task": task,
            "model_path": model_path,
            "normalized_model_path": self._normalize_model_path(model_path),
        }

    def _normalize_model_path(self, model_path):
        """Normalize a model path for comparisons and cache keys."""
        if not model_path:
            return ""

        try:
            return os.path.normcase(os.path.abspath(model_path))
        except Exception:
            return os.path.normcase(os.path.normpath(model_path))

    def _get_loaded_yolo_models(self):
        """Return the currently loaded YOLO deploy models from MainWindow."""
        accessor = getattr(self.main_window, 'get_loaded_yolo_models', None)
        if not callable(accessor):
            return []

        try:
            return accessor() or []
        except Exception:
            return []

    def _resolve_live_yolo_model(self, model_name):
        """Resolve a live YOLO source to the in-memory model object."""
        if not is_live_yolo_model(model_name):
            return None

        for source in self._get_loaded_yolo_models():
            if source.get('source_key') == model_name:
                return source.get('model')

        return None

    def _get_model_cache_key(self, model_name):
        """Build a stable cache key for the selected model source."""
        live_source = self._decode_live_yolo_source(model_name)
        if live_source:
            cache_token = (
                f"{live_source['dialog_key']}|{live_source['task']}|"
                f"{live_source['normalized_model_path']}"
            )
            digest = hashlib.sha1(cache_token.encode('utf-8')).hexdigest()[:16]
            return f"live_{live_source['dialog_key']}_{digest}"

        if os.path.sep in model_name or '/' in model_name:
            sanitized_model_name = os.path.basename(model_name)
        else:
            sanitized_model_name = model_name

        sanitized_model_name = sanitized_model_name.replace(' ', '_').replace('/', '_')
        return sanitized_model_name
    
    def _get_embedding_parameters(self):
        """Get current embedding parameters from UI."""
        technique = self.technique_combo.currentText()

        perform_pca_before = True
        if self.perform_pca_before_combo is not None:
            perform_pca_before = self.perform_pca_before_combo.currentText() == "True"

        pca_components = 50
        if self.pca_components_spin is not None:
            pca_components = self.pca_components_spin.value()

        params = {
            'technique': technique,
            'dimensions': 3 if self.dimensions_combo.currentText() == "3D" else 2,
            'perform_pca_before': perform_pca_before,
            'pca_components': pca_components,
        }

        if technique == 'PCA':
            params['perform_pca_before'] = False
        elif technique == 'UMAP':
            params['n_neighbors'] = self.umap_n_neighbors_slider.value() if self.umap_n_neighbors_slider is not None else 15
            params['min_dist'] = self.umap_min_dist_slider.value() / 100.0 if self.umap_min_dist_slider is not None else 0.1
        elif technique == 'TSNE':
            params['perplexity'] = self.tsne_perplexity_slider.value() if self.tsne_perplexity_slider is not None else 30
            params['early_exaggeration'] = self.tsne_exaggeration_slider.value() / 10.0 if self.tsne_exaggeration_slider is not None else 12.0

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
    
    def _extract_features(self, data_items, model_name, progress_bar=None, live_model=None):
        """Dispatch to appropriate feature extraction method."""
        if model_name == "Color Features":
            return self._extract_color_features(data_items, progress_bar)
        elif is_yolo_model(model_name):
            return self._extract_yolo_features(data_items, model_name, progress_bar, live_model=live_model)
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
    
    def _extract_yolo_features(self, data_items, model_name, progress_bar=None, live_model=None):
        """Extract features using YOLO model."""
        model = self._load_yolo_model(model_name, live_model=live_model)
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
    
    def _load_yolo_model(self, model_name, live_model=None):
        """Load YOLO model with caching."""
        live_source = self._decode_live_yolo_source(model_name)
        cache_allowed = live_source is None

        if live_source and live_model is not None:
            return live_model

        if live_source:
            model_name = live_source['model_path']

        if cache_allowed and self._cached_yolo_model_name == model_name and self._cached_yolo_model:
            return self._cached_yolo_model
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)
            if cache_allowed:
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
                # LDA can produce at most (n_classes - 1) components. Compute that cap.
                max_lda_components = max(1, len(set(labels)) - 1)
                n_components_lda = min(n_components, max_lda_components)
                reducer = LDA(n_components=n_components_lda)
                reducer.fit(labeled_features, labels)
                lda_transformed = reducer.transform(features_scaled)

                # If LDA produced fewer components than requested, pad the remainder
                # using PCA on the original (scaled) features so callers can still
                # request e.g. 3D output even when LDA intrinsically yields <3 dims.
                if lda_transformed.shape[1] < n_components:
                    needed = n_components - lda_transformed.shape[1]
                    # Ensure we don't ask PCA for more components than available
                    available_dim = min(needed, features_scaled.shape[1])
                    if available_dim > 0:
                        pca = PCA(n_components=available_dim, random_state=42)
                        pca_components = pca.fit_transform(features_scaled)
                        # Concatenate LDA components with PCA extras
                        combined = np.hstack([lda_transformed, pca_components[:, :available_dim]])
                        return combined
                    else:
                        return lda_transformed

                return lda_transformed
            
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

        # Ensure numpy array of shape (N, D). Convert 1D -> (N,1) to simplify downstream logic.
        embedded_features = np.asarray(embedded_features)
        if embedded_features.ndim == 1:
            embedded_features = embedded_features.reshape(-1, 1)

        n_dims = embedded_features.shape[1]
        scale_factor = 4000
        min_vals = np.min(embedded_features, axis=0)
        max_vals = np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        for i, item in enumerate(data_items):
            norm_coords = (embedded_features[i] - min_vals) / range_vals
            scaled_coords = (norm_coords * scale_factor) - (scale_factor / 2)
            
            if n_dims >= 3:
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = scaled_coords[2]
            elif n_dims == 2:
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = 0.0
            else:  # n_dims == 1
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = 0.0
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
        """
        Qt's QGraphicsScene natively uses a highly optimized C++ BSP tree 
        to instantly cull off-screen items before rendering. 
        Manually looping through thousands of points to call .setVisible() 
        actively fights the engine and causes massive lag!
        """
        pass  # Let the native C++ engine do its job!

    def _should_modify_cache(self):
        """Return True when this viewer should perform persistent cache mutations.

        Low-risk heuristic: only allow cache mutations when the viewer is visible.
        """
        try:
            return bool(self.cache_manager and self.isVisible())
        except Exception:
            return False
    
    # -------------------------------------------------------------------------
    # Selection Management
    # -------------------------------------------------------------------------
    
    def render_selection_from_ids(self, selected_ids):
        """Update visual selection using set-diffing to minimize updates."""
        blocker = QSignalBlocker(self.graphics_scene)
        try:
            selected_ids_set = set(selected_ids) if selected_ids else set()
            current_selected_ids = {aid for aid, pt in self.points_by_id.items() if pt.data_item.is_selected}

            to_select = selected_ids_set - current_selected_ids
            to_deselect = current_selected_ids - selected_ids_set

            for ann_id in to_select:
                if ann_id in self.points_by_id:
                    pt = self.points_by_id[ann_id]
                    pt.data_item.set_selected(True)
                    pt.setSelected(True)

            for ann_id in to_deselect:
                if ann_id in self.points_by_id:
                    pt = self.points_by_id[ann_id]
                    pt.data_item.set_selected(False)
                    pt.setSelected(False)
        finally:
            blocker.unblock()

        # Update internal selection state without re-iterating all points.
        # We already applied the minimal set-diff changes above, so just update
        # the cached `previous_selection_ids` and emit a consolidated signal.
        self.previous_selection_ids = set(selected_ids) if selected_ids else set()
        # Notify SelectionManager / other listeners about the new selection
        try:
            self.selection_changed.emit(list(self.previous_selection_ids))
        except Exception:
            pass

        # Lightweight UI updates
        self._update_toolbar_state()
        self._schedule_view_update()
    
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
            # Only update the points that changed to avoid O(N) work on every event
            to_select = current_ids - self.previous_selection_ids
            to_deselect = self.previous_selection_ids - current_ids

            for ann_id in to_select:
                if ann_id in self.points_by_id:
                    pt = self.points_by_id[ann_id]
                    pt.data_item.set_selected(True)
                    pt.setSelected(True)

            for ann_id in to_deselect:
                if ann_id in self.points_by_id:
                    pt = self.points_by_id[ann_id]
                    pt.data_item.set_selected(False)
                    pt.setSelected(False)

            self.previous_selection_ids = set(current_ids)
            self.selection_changed.emit(list(current_ids))

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
            # --- Explicitly make all points visible again ---
            for point in self.points_by_id.values():
                point.setVisible(True)
            # ------------------------------------------------
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
        
        pen = QPen(QColor(0, 168, 230), 2, Qt.DashLine)
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
            # Do not call _on_selection_changed() on every mouse-move while dragging;
            # emit final selection once on mouse release instead for performance.
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
            # Process the final selection exactly once
            try:
                self._on_selection_changed()
            except Exception:
                pass
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
        """Handle mouse wheel for zooming or point/sprite resizing when Ctrl is held."""
        # If Ctrl is pressed, adjust point/sprite sizes instead of zooming
        try:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta == 0:
                    return
                # Decide whether to resize sprites (when in sprites mode) or points
                if self.display_mode == 'sprites':
                    step = self._resize_step_sprite if delta > 0 else -self._resize_step_sprite
                    new_size = self.sprite_size + step
                    new_size = max(self._sprite_min, min(self._sprite_max, new_size))
                    if new_size != self.sprite_size:
                        self.sprite_size = new_size
                        # Update all point items to use new sprite size
                        for pt in self.points_by_id.values():
                            pt.update()
                        self.graphics_scene.update()
                else:
                    step = self._resize_step_point if delta > 0 else -self._resize_step_point
                    new_size = self.point_size + step
                    new_size = max(self._point_min, min(self._point_max, new_size))
                    if new_size != self.point_size:
                        self.point_size = new_size
                        for pt in self.points_by_id.values():
                            pt.update()
                        self.graphics_scene.update()
                return
        except Exception:
            return

        # Default behavior: zoom
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)

        old_pos = self.graphics_view.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.graphics_view.scale(zoom_factor, zoom_factor)

    def _key_press_event(self, event):
        """Handle key press events for the graphics view."""
        try:
            if event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
                if not self.points_by_id:
                    return

                # Grab IDs for all points that are currently visible
                ids_to_select = [
                    ann_id for ann_id, pt in self.points_by_id.items()
                    if pt.isVisible()
                ]

                if ids_to_select:
                    # Respect existing selection rendering path
                    self.render_selection_from_ids(set(ids_to_select))

                event.accept()
                return
        except Exception:
            # Fall through to default handling on error
            pass

        # Default behavior: call the native handler and refresh view
        QGraphicsView.keyPressEvent(self.graphics_view, event)

        if self.locate_graphics_item:
            # refresh location indicator positions
            QTimer.singleShot(0, self._update_location_lines)

        self._schedule_view_update()
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    @pyqtSlot()
    def clear_view(self):
        """Clear the embedding view: cancel pipeline, clear points and reset placeholders."""
        try:
            # Cancel running pipeline worker if any
            try:
                if getattr(self, '_pipeline_running', False) and getattr(self, '_pipeline_worker', None):
                    try:
                        self._pipeline_worker.cancel()
                    except Exception:
                        pass
                    try:
                        self._pipeline_worker.wait(1000)
                    except Exception:
                        pass
            except Exception:
                pass

            # Clear scene and internal state
            try:
                self._clear_points()
            except Exception:
                pass

            self.current_data_items = []
            self.current_features = None
            self.current_feature_model_key = None
            self.data_item_cache.clear()
            self.working_set_ids = []
            try:
                self.graphics_scene.clearSelection()
            except Exception:
                pass

            # Show placeholder and update toolbar
            try:
                self._show_placeholder()
            except Exception:
                pass
            try:
                self._update_toolbar_state()
            except Exception:
                pass
        except Exception:
            pass

    def closeEvent(self, event):
        """Handle close event - cleanup resources."""
        self.cache_manager.close()
        super().closeEvent(event)
