import warnings

import os

import numpy as np
import torch
import faiss

from ultralytics import YOLO

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSlot
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
                             QMainWindow, QSplitter, QGroupBox, QMessageBox,
                             QApplication, QToolBox)

from coralnet_toolbox.Explorer.QtViewers import AnnotationViewer
from coralnet_toolbox.Explorer.QtViewers import EmbeddingViewer
from coralnet_toolbox.Explorer.QtFeatureStore import FeatureStore
from coralnet_toolbox.Explorer.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.QtDataItem import EmbeddingPointItem
from coralnet_toolbox.Explorer.QtSettingsWidgets import ModelSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import EmbeddingSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import AnnotationSettingsWidget
from coralnet_toolbox.Explorer.QtAutoAnnotationWizard import AutoAnnotationWizard, AutoAnnotationError

from coralnet_toolbox.Explorer.yolo_models import is_yolo_model
from coralnet_toolbox.Explorer.transformer_models import is_transformer_model

from coralnet_toolbox.utilities import pixmap_to_numpy
from coralnet_toolbox.utilities import pixmap_to_pil

from coralnet_toolbox.Icons import get_icon

from coralnet_toolbox.QtProgressBar import ProgressBar

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from umap import UMAP
except ImportError:
    print("Warning: sklearn or umap not installed. Some features may be unavailable.")
    StandardScaler = None
    PCA = None
    TSNE = None
    LDA = None
    UMAP = None


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_WIDTH = 3
REVIEW_LABEL = 'Review'

# ----------------------------------------------------------------------------------------------------------------------
# ExplorerWindow
# ----------------------------------------------------------------------------------------------------------------------


class ExplorerWindow(QMainWindow):
    def __init__(self, main_window, parent=None):
        """Initialize the ExplorerWindow."""
        super(ExplorerWindow, self).__init__(parent)
        self.main_window = main_window
        self.image_window = main_window.image_window
        self.label_window = main_window.label_window
        self.annotation_window = main_window.annotation_window
        
        self.animation_manager = main_window.animation_manager

        self.device = main_window.device
        self.loaded_model = None
        self.imgsz = 128

        self.feature_store = FeatureStore()
        
        # Add a property to store the parameters with defaults
        self.anomaly_params = {'contamination': 0.1, 'n_neighbors': 20, 'threshold': 75}
        self.similarity_params = {'k': 50, 'same_label': False, 'same_image': False, 'min_confidence': 0.0}
        self.duplicate_params = {'threshold': 0.5, 'same_image': True, 'spatial_threshold': 100}
        
        self.data_item_cache = {}  # Cache for AnnotationDataItem objects

        self.current_data_items = []
        self.current_features = None
        self.current_feature_generating_model = ""
        self.current_embedding_model_info = None
        self._ui_initialized = False

        self.setWindowTitle("Explorer")
        explorer_icon_path = get_icon("magic.png")
        self.setWindowIcon(QIcon(explorer_icon_path))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.left_panel = QWidget()
        self.label_layout = QVBoxLayout(self.left_panel)

        self.annotation_settings_widget = None
        self.model_settings_widget = None
        self.embedding_settings_widget = None
        self.annotation_viewer = None
        self.embedding_viewer = None
        
        self.auto_annotation_button = QPushButton('🤖 Auto Annotation Wizard', self)
        self.auto_annotation_button.clicked.connect(self.open_auto_annotation_wizard)
        self.auto_annotation_button.setToolTip("Open the ML-assisted annotation wizard (requires features/embeddings)")
        self.auto_annotation_button.setEnabled(False)
        
        self.auto_annotation_wizard = None  # Will be created when needed

        self.clear_preview_button = QPushButton('Clear Preview', self)
        self.clear_preview_button.clicked.connect(self.clear_preview_changes)
        self.clear_preview_button.setToolTip("Clear all preview changes and revert to original labels")
        self.clear_preview_button.setEnabled(False)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setToolTip("Close the window")

        self.apply_button = QPushButton('Apply', self)
        self.apply_button.clicked.connect(self.apply)
        self.apply_button.setToolTip("Apply changes")
        self.apply_button.setEnabled(False)

    def showEvent(self, event):
        """Handle show event."""
        if not self._ui_initialized:
            self.setup_ui()
            self._ui_initialized = True
        super(ExplorerWindow, self).showEvent(event)

    def closeEvent(self, event):
        """Handle close event."""
        # Stop any running timers to prevent errors
        if hasattr(self, 'embedding_viewer') and self.embedding_viewer:
            if hasattr(self.embedding_viewer, 'animation_timer') and self.embedding_viewer.animation_timer:
                self.embedding_viewer.animation_timer.stop()

        # Call the main cancellation method to revert any pending changes and clear selections.
        self.clear_preview_changes()

        # Clean up the feature store by deleting its files
        if hasattr(self, 'feature_store') and self.feature_store:
            self.feature_store.delete_storage()

        # Call the dedicated cleanup method
        self._cleanup_resources()

        # Re-enable the main window before closing
        if self.main_window:
            self.main_window.setEnabled(True)

        # Move the label_window back to the main_window
        if hasattr(self.main_window, 'explorer_closed'):
            self.main_window.explorer_closed()

        # Clear the reference in the main_window to allow garbage collection
        self.main_window.explorer_window = None

        # Set the ui_initialized flag to False so it can be re-initialized next time
        self._ui_initialized = False

        event.accept()

    def setup_ui(self):
        """Set up the UI for the ExplorerWindow."""
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)

        # Lazily initialize the settings and viewer widgets
        if self.annotation_settings_widget is None:
            self.annotation_settings_widget = AnnotationSettingsWidget(self.main_window, self)
        if self.model_settings_widget is None:
            self.model_settings_widget = ModelSettingsWidget(self.main_window, self)
        if self.embedding_settings_widget is None:
            self.embedding_settings_widget = EmbeddingSettingsWidget(self.main_window, self)
        if self.annotation_viewer is None:
            self.annotation_viewer = AnnotationViewer(self)
        if self.embedding_viewer is None:
            self.embedding_viewer = EmbeddingViewer(self)

        # Vertical settings panel on the far left is now a QToolBox
        settings_toolbox = QToolBox()
        settings_toolbox.addItem(self.annotation_settings_widget, "1. Annotation Filters")
        settings_toolbox.addItem(self.model_settings_widget, "2. Model Selection")
        settings_toolbox.addItem(self.embedding_settings_widget, "3. Embedding Parameters")
        
        # Horizontal splitter for the two main viewer panels
        middle_splitter = QSplitter(Qt.Horizontal)
        annotation_group = QGroupBox("Annotation Viewer")
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.addWidget(self.annotation_viewer)
        middle_splitter.addWidget(annotation_group)

        embedding_group = QGroupBox("Embedding Viewer")
        embedding_layout = QVBoxLayout(embedding_group)
        embedding_layout.addWidget(self.embedding_viewer)
        middle_splitter.addWidget(embedding_group)
        middle_splitter.setSizes([500, 500])

        # Left panel: Reuse existing if it has the LabelWindow, otherwise create new
        if not hasattr(self, 'left_panel') or not self.left_panel:
            self.left_panel = QWidget()
            self.label_layout = QVBoxLayout(self.left_panel)
        
        # Add the LabelWindow above the settings toolbox
        self.label_layout.addWidget(self.label_window)
        # Add the settings toolbox below the label_layout
        self.label_layout.addWidget(settings_toolbox)
        
        # Set fixed width for left_panel to keep it always visible and non-resizable
        self.left_panel.setFixedWidth(300)

        # Create horizontal layout for left panel and viewers (no splitter for fixed left panel)
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.left_panel)
        horizontal_layout.addWidget(middle_splitter)

        self.main_layout.addLayout(horizontal_layout)

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.addWidget(self.auto_annotation_button)
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.clear_preview_button)
        self.buttons_layout.addWidget(self.exit_button)
        self.buttons_layout.addWidget(self.apply_button)
        self.main_layout.addLayout(self.buttons_layout)
        
        self._initialize_data_item_cache()
        self.annotation_settings_widget.set_default_to_current_image()
        self.refresh_filters()

        try:
            self.label_window.labelSelected.disconnect(self.on_label_selected_for_preview)
        except TypeError:
            pass

        # Connect signals to slots
        self.annotation_window.annotationModified.connect(self.on_annotation_modified)
        self.label_window.labelSelected.connect(self.on_label_selected_for_preview)
        self.annotation_viewer.selection_changed.connect(self.on_annotation_view_selection_changed)
        self.annotation_viewer.preview_changed.connect(self.on_preview_changed)
        self.annotation_viewer.reset_view_requested.connect(self.on_reset_view_requested)
        self.embedding_viewer.selection_changed.connect(self.on_embedding_view_selection_changed)
        self.embedding_viewer.reset_view_requested.connect(self.on_reset_view_requested)
        self.embedding_viewer.find_anomalies_requested.connect(self.find_anomalies)
        self.embedding_viewer.anomaly_parameters_changed.connect(self.on_anomaly_params_changed)
        self.model_settings_widget.selection_changed.connect(self.on_model_selection_changed)
        self.embedding_viewer.find_duplicates_requested.connect(self.find_duplicate_annotations)
        self.embedding_viewer.duplicate_parameters_changed.connect(self.on_duplicate_params_changed)
        self.embedding_viewer.find_similar_requested.connect(self.find_similar_annotations)
        self.embedding_viewer.similarity_parameters_changed.connect(self.on_similarity_params_changed)
        
    def _clear_selections(self):
        """Clears selections in both viewers and stops animations."""
        if not self._ui_initialized:
            return
            
        # Clear selection in the annotation viewer, which also stops widget animations.
        if self.annotation_viewer:
            self.annotation_viewer.clear_selection()

        # Clear selection in the embedding viewer. This deselects all points
        # and stops the animation timer via its on_selection_changed handler.
        if self.embedding_viewer:
            self.embedding_viewer.render_selection_from_ids(set())

        # Update other UI elements that depend on selection state.
        self.update_label_window_selection()
        self.update_button_states()
        
        # Process events
        QApplication.processEvents()
        print("Cleared all active selections.")
        
    @pyqtSlot(list)
    def on_annotation_view_selection_changed(self, changed_ann_ids):
        """Syncs selection from AnnotationViewer to other components and manages UI state."""
        # Unselect any annotation in the main AnnotationWindow for a clean slate
        if hasattr(self, 'annotation_window'):
            self.annotation_window.unselect_annotations()

        all_selected_ids = {w.data_item.annotation.id for w in self.annotation_viewer.selected_widgets}

        # Sync selection to the embedding viewer
        if self.embedding_viewer.points_by_id:
            blocker = QSignalBlocker(self.embedding_viewer)
            self.embedding_viewer.render_selection_from_ids(all_selected_ids)

        # Get the select tool to manage its state
        select_tool = self.annotation_window.tools.get('select')
        if select_tool:
            # If the selection from the explorer is not a single item (i.e., it's empty
            # or a multi-selection), hide the handles and release the lock.
            if len(all_selected_ids) != 1:
                select_tool._hide_resize_handles()
                select_tool.selection_locked = False
        # Ensure that the select tool is not active     
        self.annotation_window.set_selected_tool(None)

        # Update the label window based on the new selection
        self.update_label_window_selection()

    @pyqtSlot(list)
    def on_embedding_view_selection_changed(self, all_selected_ann_ids):
        """Syncs selection from EmbeddingViewer to AnnotationViewer and isolates."""
        selected_ids_set = set(all_selected_ann_ids)

        # If a selection is made in the embedding viewer, isolate those widgets.
        if selected_ids_set:
            # This new method will handle setting the isolated set and selecting them.
            self.annotation_viewer.isolate_and_select_from_ids(selected_ids_set)
        # If the selection is cleared in the embedding viewer, exit isolation mode.
        elif self.annotation_viewer.isolated_mode:
            self.annotation_viewer.show_all_annotations()

        # We still need to update the label window based on the selection.
        self.update_label_window_selection()

    @pyqtSlot(list)
    def on_preview_changed(self, changed_ann_ids):
        """Updates embedding point colors and tooltips when a preview label is applied."""
        for ann_id in changed_ann_ids:
            # Update embedding point color
            point = self.embedding_viewer.points_by_id.get(ann_id)
            if point:
                point.update()
                point.update_tooltip()  # Refresh tooltip to show new effective label

            # Update annotation widget tooltip
            widget = self.annotation_viewer.annotation_widgets_by_id.get(ann_id)
            if widget:
                widget.update_tooltip()
                
    @pyqtSlot(str)
    def on_annotation_modified(self, annotation_id):
        """
        Handles an annotation being moved or resized in the AnnotationWindow.
        This invalidates the cached features and updates the annotation's thumbnail.
        """
        print(f"Annotation {annotation_id} was modified. Removing its cached features.")
        if hasattr(self, 'feature_store'):
            # This method must exist on the FeatureStore to clear features
            # for the given annotation ID across all stored models.
            self.feature_store.remove_features_for_annotation(annotation_id)
            
        # Update the AnnotationImageWidget in the AnnotationViewer
        if hasattr(self, 'annotation_viewer'):
            # Find the corresponding widget by its annotation ID
            widget = self.annotation_viewer.annotation_widgets_by_id.get(annotation_id)
            if widget:
                # The annotation's geometry may have changed, so we need to update the widget.
                # 1. Recalculate the aspect ratio.
                widget.recalculate_aspect_ratio()
                # 2. Unload the stale image data. This marks the widget as "dirty".
                widget.unload_image()
                # 3. Recalculate the layout. This will resize the widget based on the new
                #    aspect ratio and reload the image if the widget is currently visible.
                self.annotation_viewer.recalculate_layout()

    @pyqtSlot()
    def on_reset_view_requested(self):
        """Handle reset view requests from double-click in either viewer."""
        # Clear all selections in both viewers
        self.annotation_viewer.clear_selection()
        self.embedding_viewer.render_selection_from_ids(set())

        # Exit isolation mode if currently active in AnnotationViewer
        if self.annotation_viewer.isolated_mode:
            self.annotation_viewer.show_all_annotations()

        if self.embedding_viewer.isolated_mode:
            self.embedding_viewer.show_all_points()

        # Clear similarity sort context
        self.annotation_viewer.active_ordered_ids = []

        self.update_label_window_selection()
        self.update_button_states()

        print("Reset view: cleared selections and exited isolation mode")
        
    @pyqtSlot(dict)
    def on_anomaly_params_changed(self, params):
        """Updates the stored parameters for anomaly detection."""
        self.anomaly_params = params
        print(f"Anomaly detection parameters updated: {self.anomaly_params}")
        
    @pyqtSlot(dict)
    def on_duplicate_params_changed(self, params):
        """Updates the stored parameters for duplicate detection."""
        self.duplicate_params = params
        print(f"Duplicate detection parameters updated: {self.duplicate_params}")
        
    @pyqtSlot(dict)
    def on_similarity_params_changed(self, params):
        """Updates the stored parameters for similarity search."""
        self.similarity_params = params
        print(f"Similarity search parameters updated: {self.similarity_params}")
        
    @pyqtSlot()
    def on_model_selection_changed(self):
        """
        Handles changes in the model settings to enable/disable model-dependent features.
        """
        if not self._ui_initialized:
            return

        model_name, feature_mode = self.model_settings_widget.get_selected_model()
        
        # Update toolbar state when model changes
        self.embedding_viewer._update_toolbar_state()
        
    def _initialize_data_item_cache(self):
        """
        Creates a persistent AnnotationDataItem for every annotation,
        caching them for the duration of the session.
        """
        self.data_item_cache.clear()
        if not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return

        all_annotations = self.main_window.annotation_window.annotations_dict.values()
        for ann in all_annotations:
            if ann.id not in self.data_item_cache:
                self.data_item_cache[ann.id] = AnnotationDataItem(ann)

    def update_label_window_selection(self):
        """
        Updates the label window based on the selection state of the currently
        loaded data items. This is the single, centralized point of logic.
        """
        # Get selected items directly from the master data list
        selected_data_items = [
            item for item in self.current_data_items if item.is_selected
        ]

        if not selected_data_items:
            self.label_window.deselect_active_label()
            self.label_window.update_annotation_count()
            return

        first_effective_label = selected_data_items[0].effective_label
        all_same_current_label = all(
            item.effective_label.id == first_effective_label.id
            for item in selected_data_items
        )

        if all_same_current_label:
            # Get the actual Label widget from LabelWindow by ID to ensure object identity
            label_widget = self.label_window.get_label_by_id(first_effective_label.id, return_review=True)
            if label_widget:
                self.label_window.set_active_label(label_widget)
                # This emit is what updates other UI elements, like the annotation list
                self.annotation_window.labelSelected.emit(first_effective_label.id)
            else:
                self.label_window.deselect_active_label()
        else:
            self.label_window.deselect_active_label()

        self.label_window.update_annotation_count()

    def get_filtered_data_items(self):
        """
        Gets annotations matching all conditions by retrieving their
        persistent AnnotationDataItem objects from the cache.
        """
        if not hasattr(self.main_window.annotation_window, 'annotations_dict'):
            return []

        selected_images = self.annotation_settings_widget.get_selected_images()
        selected_types = self.annotation_settings_widget.get_selected_annotation_types()
        selected_labels = self.annotation_settings_widget.get_selected_labels()

        if not all([selected_images, selected_types, selected_labels]):
            return []

        annotations_to_process = [
            ann for ann in self.main_window.annotation_window.annotations_dict.values()
            if (os.path.basename(ann.image_path) in selected_images and
                type(ann).__name__ in selected_types and
                ann.label.short_label_code in selected_labels)
        ]

        self._ensure_cropped_images(annotations_to_process)
        
        return [self.data_item_cache[ann.id] for ann in annotations_to_process if ann.id in self.data_item_cache]
    
    def find_anomalies(self):
        """
        Identifies anomalous annotations using Local Outlier Factor (LOF) and Isolation Forest.
        Computes anomaly scores for each annotation and selects those exceeding threshold.
        Also calculates quality scores based on local density and spatial consistency.
        """
        # Get parameters
        contamination = self.anomaly_params.get('contamination', 0.1)
        n_neighbors = self.anomaly_params.get('n_neighbors', 20)
        threshold_percentile = self.anomaly_params.get('threshold', 75)

        if not self.embedding_viewer.points_by_id or len(self.embedding_viewer.points_by_id) < n_neighbors:
            QMessageBox.information(self, 
                                    "Not Enough Data",
                                    f"This feature requires at least {n_neighbors} points in the embedding viewer.")
            return

        items_in_view = list(self.embedding_viewer.points_by_id.values())
        data_items_in_view = [p.data_item for p in items_in_view]

        # Get the model key used for the current embedding
        model_info = self.model_settings_widget.get_selected_model()
        model_name, feature_mode = model_info if isinstance(model_info, tuple) else (model_info, "default")
        sanitized_model_name = os.path.basename(model_name).replace(' ', '_')
        sanitized_feature_mode = feature_mode.replace(' ', '_').replace('/', '_')
        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.ensemble import IsolationForest
            
            # Get the high-dimensional features
            features_dict, _ = self.feature_store.get_features(data_items_in_view, model_key)
            if not features_dict:
                QMessageBox.warning(self, 
                                    "Error", 
                                    "Could not retrieve features for the items in view.")
                return

            query_ann_ids = list(features_dict.keys())
            query_vectors = np.array([features_dict[ann_id] for ann_id in query_ann_ids]).astype('float32')

            # 1. Local Outlier Factor (LOF) - detects local density anomalies
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
            lof_predictions = lof.fit_predict(query_vectors)
            lof_scores = -lof.negative_outlier_factor_  # Convert to positive scores (higher = more anomalous)
            
            # 2. Isolation Forest - detects global isolation anomalies
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_predictions = iso_forest.fit_predict(query_vectors)
            iso_scores = -iso_forest.score_samples(query_vectors)  # Convert to positive scores
            
            # Normalize scores to 0-1 range
            lof_scores_normalized = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
            iso_scores_normalized = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)
            
            # Combine scores (weighted average)
            combined_scores = 0.6 * lof_scores_normalized + 0.4 * iso_scores_normalized
            
            # Calculate local density for quality scoring
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(query_vectors)
            distances, indices = nn.kneighbors(query_vectors)
            
            # Average distance to k nearest neighbors (excluding self)
            avg_distances = distances[:, 1:].mean(axis=1)
            local_densities = 1.0 / (avg_distances + 1e-10)  # Inverse distance = density
            
            # Calculate spatial consistency (label agreement with neighbors)
            spatial_consistencies = []
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                label_id = data_item.effective_label.id
                
                # Check neighbor labels
                neighbor_indices = indices[i][1:]  # Exclude self
                matching_neighbors = 0
                valid_neighbors = 0
                
                for neighbor_idx in neighbor_indices:
                    neighbor_ann_id = query_ann_ids[neighbor_idx]
                    neighbor_item = self.data_item_cache[neighbor_ann_id]
                    
                    # Only count neighbors from the same image for spatial consistency
                    if neighbor_item.annotation.image_path == data_item.annotation.image_path:
                        valid_neighbors += 1
                        if neighbor_item.effective_label.id == label_id:
                            matching_neighbors += 1
                
                if valid_neighbors > 0:
                    consistency = matching_neighbors / valid_neighbors
                else:
                    consistency = 0.5  # Neutral if no same-image neighbors
                
                spatial_consistencies.append(consistency)
            
            spatial_consistencies = np.array(spatial_consistencies)
            
            # Update data items with scores
            anomaly_threshold = np.percentile(combined_scores, threshold_percentile)
            anomalous_ann_ids = []
            
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                
                # Store anomaly and quality metrics
                data_item.anomaly_score = float(combined_scores[i])
                data_item.local_density = float(local_densities[i])
                data_item.spatial_consistency = float(spatial_consistencies[i])
                
                # Calculate quality score
                data_item.calculate_quality_score()
                
                # Update tooltip to show new info
                if hasattr(data_item, 'point_item') and data_item.point_item:
                    data_item.point_item.update_tooltip()
                
                # Select if anomaly score exceeds threshold
                if combined_scores[i] >= anomaly_threshold:
                    anomalous_ann_ids.append(ann_id)
            
            # Select anomalous items and sort by anomaly score (most anomalous first)
            sorted_anomalous_ids = sorted(anomalous_ann_ids, 
                                         key=lambda aid: self.data_item_cache[aid].anomaly_score, 
                                         reverse=True)
            
            self.annotation_viewer.display_and_isolate_ordered_results(sorted_anomalous_ids)
            self.embedding_viewer.render_selection_from_ids(set(sorted_anomalous_ids))
            
            # Show summary message
            total_items = len(query_ann_ids)
            num_anomalies = len(anomalous_ann_ids)
            QMessageBox.information(self,
                                   "Anomaly Detection Complete",
                                   f"Found {num_anomalies} anomalous annotations out of {total_items} total.\n\n"
                                   f"Threshold: {threshold_percentile}th percentile (score ≥ {anomaly_threshold:.3f})\n"
                                   f"Results sorted by anomaly score (highest first).")

        except ImportError as e:
            QMessageBox.critical(self,
                               "Missing Dependencies",
                               f"Anomaly detection requires scikit-learn.\n\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def find_duplicate_annotations(self):
        """
        Enhanced duplicate detection with multi-stage filtering:
        1. Visual similarity (feature distance)
        2. Spatial proximity (same image + nearby locations)
        3. Metadata consistency (if available)
        
        Uses DSU to group transitively connected duplicates.
        """
        threshold = self.duplicate_params.get('threshold', 0.05)
        same_image_only = self.duplicate_params.get('same_image', True)
        spatial_threshold = self.duplicate_params.get('spatial_threshold', 100)  # pixels

        if not self.embedding_viewer.points_by_id or len(self.embedding_viewer.points_by_id) < 2:
            QMessageBox.information(self, 
                                    "Not Enough Data", 
                                    "This feature requires at least 2 points in the embedding viewer.")
            return

        items_in_view = list(self.embedding_viewer.points_by_id.values())
        data_items_in_view = [p.data_item for p in items_in_view]

        model_info = self.model_settings_widget.get_selected_model()
        model_name, feature_mode = model_info if isinstance(model_info, tuple) else (model_info, "default")
        sanitized_model_name = os.path.basename(model_name).replace(' ', '_')
        sanitized_feature_mode = feature_mode.replace(' ', '_').replace('/', '_')
        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            index = self.feature_store._get_or_load_index(model_key)
            if index is None:
                QMessageBox.warning(self, "Error", "Could not find a valid feature index for the current model.")
                return

            features_dict, _ = self.feature_store.get_features(data_items_in_view, model_key)
            if not features_dict:
                QMessageBox.warning(self, "Error", "Could not retrieve features for the items in view.")
                return

            query_ann_ids = list(features_dict.keys())
            query_vectors = np.array([features_dict[ann_id] for ann_id in query_ann_ids]).astype('float32')

            # Stage 1: Find visually similar candidates (feature similarity)
            # Search for more neighbors to catch potential duplicates
            k_search = min(10, len(query_ann_ids))
            D, I = index.search(query_vectors, k_search)

            # Use a Disjoint Set Union (DSU) data structure to group duplicates
            parent = {ann_id: ann_id for ann_id in query_ann_ids}
            
            def find_set(v):
                if v == parent[v]:
                    return v
                parent[v] = find_set(parent[v])
                return parent[v]
            
            def unite_sets(a, b):
                a = find_set(a)
                b = find_set(b)
                if a != b:
                    parent[b] = a
            
            id_map = self.feature_store.get_faiss_index_to_annotation_id_map(model_key)
            
            # Helper function to calculate spatial distance between annotations
            def get_spatial_distance(item1, item2):
                """Calculate Euclidean distance between annotation centers."""
                try:
                    # Get bounding box centers
                    tl1 = item1.annotation.get_bounding_box_top_left()
                    br1 = item1.annotation.get_bounding_box_bottom_right()
                    tl2 = item2.annotation.get_bounding_box_top_left()
                    br2 = item2.annotation.get_bounding_box_bottom_right()
                    
                    if tl1 and br1 and tl2 and br2:
                        center1_x = (tl1.x() + br1.x()) / 2
                        center1_y = (tl1.y() + br1.y()) / 2
                        center2_x = (tl2.x() + br2.x()) / 2
                        center2_y = (tl2.y() + br2.y()) / 2
                        
                        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
                        return distance
                except (AttributeError, TypeError):
                    pass
                return float('inf')

            # Stage 2 & 3: Filter by spatial proximity and metadata
            duplicate_pairs = []
            
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                
                # Check each neighbor (skip index 0 which is self)
                for j in range(1, k_search):
                    if j >= len(I[i]):
                        break
                        
                    neighbor_faiss_idx = I[i, j]
                    distance = D[i, j]

                    # Stage 1 check: Visual similarity
                    if distance >= threshold:
                        continue  # Not similar enough
                    
                    neighbor_ann_id = id_map.get(neighbor_faiss_idx)
                    if not neighbor_ann_id or neighbor_ann_id not in self.data_item_cache:
                        continue
                    
                    neighbor_item = self.data_item_cache[neighbor_ann_id]
                    
                    # Stage 2 check: Same image (if required)
                    if same_image_only and data_item.annotation.image_path != neighbor_item.annotation.image_path:
                        continue  # Different images - skip if same_image_only is True
                    
                    # Stage 3 check: Spatial proximity (only for same-image annotations)
                    if data_item.annotation.image_path == neighbor_item.annotation.image_path:
                        spatial_dist = get_spatial_distance(data_item, neighbor_item)
                        if spatial_dist > spatial_threshold:
                            continue  # Too far apart in same image
                    else:
                        spatial_dist = float('inf')  # Different images
                    
                    # Stage 4 check: Label consistency (same label increases confidence)
                    if data_item.effective_label.id != neighbor_item.effective_label.id:
                        # Different labels but visually similar and nearby - might be annotation error
                        # Still consider as duplicate but with lower confidence
                        pass
                    
                    # Passed all filters - mark as duplicate
                    duplicate_pairs.append((ann_id, neighbor_ann_id, distance, spatial_dist))
                    unite_sets(ann_id, neighbor_ann_id)
            
            # Group annotations by their set representative
            groups = {}
            for ann_id in query_ann_ids:
                root = find_set(ann_id)
                if root not in groups:
                    groups[root] = []
                groups[root].append(ann_id)

            # Select duplicate copies (keep one original per group)
            copies_to_select = []
            duplicate_info = []
            
            for root_id, group_ids in groups.items():
                if len(group_ids) > 1:
                    # Sort IDs to consistently pick the same "original"
                    sorted_ids = sorted(group_ids)
                    original = sorted_ids[0]
                    duplicates = sorted_ids[1:]
                    copies_to_select.extend(duplicates)
                    
                    # Collect info for summary
                    duplicate_info.append({
                        'original': original,
                        'duplicates': duplicates,
                        'count': len(group_ids)
                    })
            
            # Sort results by number of duplicates per group (most duplicates first)
            sorted_copies = sorted(copies_to_select, 
                                  key=lambda aid: next((info['count'] for info in duplicate_info 
                                                       if aid in info['duplicates']), 0),
                                  reverse=True)
            
            self.annotation_viewer.display_and_isolate_ordered_results(sorted_copies)
            self.embedding_viewer.render_selection_from_ids(set(sorted_copies))
            
            # Show summary
            num_groups = len(duplicate_info)
            num_duplicates = len(copies_to_select)
            # Build summary message
            filters_text = f"• Visual similarity threshold: {threshold}\n"
            if same_image_only:
                filters_text += f"• Same image only\n• Spatial proximity: {spatial_threshold}px\n"
            else:
                filters_text += "• Across all images\n"
            
            QMessageBox.information(self,
                                   "Duplicate Detection Complete",
                                   f"Found {num_groups} duplicate groups with {num_duplicates} duplicate copies.\n\n"
                                   f"Filters applied:\n"
                                   f"{filters_text}\n"
                                   f"Results sorted by group size (largest first).")

        finally:
            QApplication.restoreOverrideCursor()
            
    @pyqtSlot()
    def find_similar_annotations(self):
        """
        Finds k-nearest neighbors to the selected annotation(s) using cosine similarity.
        Uses confidence-weighted centroids and supports filtering by label, image, and confidence.
        """
        k = self.similarity_params.get('k', 50)
        same_label = self.similarity_params.get('same_label', False)
        same_image = self.similarity_params.get('same_image', False)
        min_confidence = self.similarity_params.get('min_confidence', 0.0)

        # Get selected items from embedding viewer
        selected_points = [point for point in self.embedding_viewer.graphics_scene.selectedItems()
                          if isinstance(point, EmbeddingPointItem)]
        
        if not selected_points:
            QMessageBox.information(self, "No Selection", 
                                  "Please select one or more points in the embedding viewer first.")
            return

        if not self.current_embedding_model_info:
            QMessageBox.warning(self, "No Embedding", 
                              "Please run an embedding before searching for similar items.")
            return

        selected_data_items = [point.data_item for point in selected_points]
        model_name, feature_mode = self.current_embedding_model_info
        sanitized_model_name = os.path.basename(model_name).replace(' ', '_')
        sanitized_feature_mode = feature_mode.replace(' ', '_').replace('/', '_')
        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Get features for selected items
            features_dict, _ = self.feature_store.get_features(selected_data_items, model_key)
            if not features_dict:
                QMessageBox.warning(self, 
                                  "Features Not Found", 
                                  "Could not retrieve feature vectors for the selected items.")
                return

            # Create confidence-weighted query vector
            source_vectors = []
            confidences = []
            for item in selected_data_items:
                if item.annotation.id in features_dict:
                    source_vectors.append(features_dict[item.annotation.id])
                    # Use effective confidence, defaulting to 1.0 if unavailable
                    conf = item.get_effective_confidence()
                    confidences.append(conf if conf is not None else 1.0)
            
            source_vectors = np.array(source_vectors).astype('float32')
            confidences = np.array(confidences)
            
            # Normalize confidences to use as weights
            if confidences.sum() > 0:
                weights = confidences / confidences.sum()
            else:
                weights = np.ones(len(confidences)) / len(confidences)
            
            # Compute weighted centroid
            query_vector = np.average(source_vectors, axis=0, weights=weights, keepdims=True).astype('float32')
            
            # Normalize query vector for cosine similarity
            faiss.normalize_L2(query_vector)

            # Get or create normalized index for cosine similarity
            index = self._get_or_create_normalized_index(model_key)
            faiss_idx_to_ann_id = self.feature_store.get_faiss_index_to_annotation_id_map(model_key)
            
            if index is None or not faiss_idx_to_ann_id:
                QMessageBox.warning(self, 
                                  "Index Error", 
                                  "Could not find a valid feature index for the current model.")
                return

            # Search for more candidates than needed to account for filtering
            num_to_find = min(k * 5, index.ntotal)  # Search 5x more for filtering
            distances, I = index.search(query_vector, num_to_find)

            # Filter and collect results
            source_ids = {item.annotation.id for item in selected_data_items}
            similar_items = []
            
            for idx, (faiss_idx, distance) in enumerate(zip(I[0], distances[0])):
                ann_id = faiss_idx_to_ann_id.get(faiss_idx)
                if not ann_id or ann_id not in self.data_item_cache or ann_id in source_ids:
                    continue
                
                data_item = self.data_item_cache[ann_id]
                
                # Apply filters
                if same_label:
                    # Check if all selected items have the same label
                    selected_labels = {item.effective_label.id for item in selected_data_items}
                    if len(selected_labels) == 1:
                        if data_item.effective_label.id not in selected_labels:
                            continue
                    else:
                        # If multiple labels selected, skip this filter
                        pass
                
                if same_image:
                    # Check if from same image as any selected item
                    selected_images = {item.annotation.image_path for item in selected_data_items}
                    if data_item.annotation.image_path not in selected_images:
                        continue
                
                if min_confidence > 0:
                    item_confidence = data_item.get_effective_confidence()
                    if item_confidence is None or item_confidence < min_confidence:
                        continue
                
                # Store similarity score (convert distance to similarity)
                # For inner product after normalization, higher score = more similar
                similarity_score = float(distance)  # Already cosine similarity
                data_item.similarity_score = similarity_score
                data_item.similarity_rank = len(similar_items) + 1
                
                similar_items.append((ann_id, similarity_score))
                
                if len(similar_items) >= k:
                    break

            if not similar_items:
                QMessageBox.information(self, 
                                      "No Results", 
                                      "No similar items found matching the filter criteria.")
                return

            # Sort by similarity (highest first) and get IDs
            similar_items.sort(key=lambda x: x[1], reverse=True)
            similar_ann_ids = [ann_id for ann_id, _ in similar_items]

            # Create ordered list: selection first, then similar items
            ordered_ids_to_display = list(source_ids) + similar_ann_ids
            
            # Update UI
            self.annotation_viewer.sort_combo.setCurrentText("None")
            self.embedding_viewer.render_selection_from_ids(set(ordered_ids_to_display))
            self.annotation_viewer.display_and_isolate_ordered_results(ordered_ids_to_display)
            
            self.update_button_states()
            
            # Show summary
            filter_text = []
            if same_label:
                filter_text.append("same label")
            if same_image:
                filter_text.append("same image")
            if min_confidence > 0:
                filter_text.append(f"min confidence {min_confidence:.0%}")
            
            filter_str = " + ".join(filter_text) if filter_text else "no filters"
            
            QMessageBox.information(self,
                                  "Similar Items Found",
                                  f"Found {len(similar_items)} similar items (requested: {k})\n"
                                  f"Filters: {filter_str}\n"
                                  f"Using cosine similarity on confidence-weighted query.")

        finally:
            QApplication.restoreOverrideCursor()
    
    def _get_or_create_normalized_index(self, model_key):
        """
        Gets or creates a normalized FAISS index for cosine similarity using IndexFlatIP.
        Caches normalized indexes separately from the standard L2 indexes.
        """
        # Check if we have a cached normalized index
        normalized_key = f"{model_key}_normalized"
        if hasattr(self, '_normalized_indexes') and normalized_key in self._normalized_indexes:
            return self._normalized_indexes[normalized_key]
        
        # Get the original index from feature store
        original_index = self.feature_store._get_or_load_index(model_key)
        if original_index is None:
            return None
        
        # Extract all vectors and normalize them
        n_vectors = original_index.ntotal
        if n_vectors == 0:
            return None
        
        vectors = original_index.reconstruct_n(0, n_vectors)
        vectors = vectors.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Create inner product index (equivalent to cosine similarity after normalization)
        dim = vectors.shape[1]
        normalized_index = faiss.IndexFlatIP(dim)
        normalized_index.add(vectors)
        
        # Cache the normalized index
        if not hasattr(self, '_normalized_indexes'):
            self._normalized_indexes = {}
        self._normalized_indexes[normalized_key] = normalized_index
        
        return normalized_index
            
    def _get_yolo_predictions_for_uncertainty(self, data_items, model_info):
        """
        Runs a YOLO classification model to get probabilities for uncertainty analysis.
        This is a streamlined method that does NOT use the feature store.
        """
        model_name, feature_mode = model_info
        
        # Load the model
        model = self._load_yolo_model(model_name, feature_mode)
        if model is None:
            QMessageBox.warning(self, 
                                "Model Load Error",
                                f"Could not load YOLO model '{model_name}'.")
            return None
        
        # Prepare images from data items with proper resizing
        image_list, valid_data_items = self._prepare_images_from_data_items(
            data_items,
            format='numpy',
            target_size=(self.imgsz, self.imgsz)
        )
        
        if not image_list:
            return None
        
        try:
            # We need probabilities for uncertainty analysis, so we always use predict
            results = model.predict(image_list, 
                                    stream=False,  # Use batch processing for uncertainty
                                    imgsz=self.imgsz, 
                                    half=True, 
                                    device=self.device, 
                                    verbose=False)
                
            _, probabilities_dict = self._process_model_results(results, valid_data_items, "Predictions")
            return probabilities_dict
            
        except TypeError:
            QMessageBox.warning(self, 
                                "Invalid Model",
                                "The selected model is not compatible with uncertainty analysis.")
            return None
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _ensure_cropped_images(self, annotations):
        """Ensures all provided annotations have a cropped image available."""
        annotations_by_image = {}

        for annotation in annotations:
            if not annotation.cropped_image:
                image_path = annotation.image_path
                if image_path not in annotations_by_image:
                    annotations_by_image[image_path] = []
                annotations_by_image[image_path].append(annotation)

        if not annotations_by_image:
            return

        progress_bar = ProgressBar(self, "Cropping Image Annotations")
        progress_bar.show()
        progress_bar.start_progress(len(annotations_by_image))

        try:
            for image_path, image_annotations in annotations_by_image.items():
                self.annotation_window.crop_annotations(image_path=image_path,
                                                        annotations=image_annotations,
                                                        return_annotations=False,
                                                        verbose=False)
                progress_bar.update_progress()
        finally:
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
            
    def _load_yolo_model(self, model_name, feature_mode):
        """
        Helper function to load a YOLO model and cache it.
        
        Args:
            model_name (str): Path to the YOLO model file
            feature_mode (str): Mode for feature extraction ("Embed Features" or "Predictions")
        
        Returns:
            ultralytics.yolo.engine.model.Model: The loaded YOLO model object, or None if loading fails.
        """
        current_run_key = (model_name, feature_mode)
        
        # Force a reload if the model path OR the feature mode has changed
        if current_run_key != self.current_feature_generating_model or self.loaded_model is None:
            print(f"Model or mode changed. Reloading {model_name} for '{feature_mode}'.")
            try:
                model = YOLO(model_name)
                
                # Check if the model task is compatible with the selected feature mode
                if model.task != 'classify' and feature_mode == "Predictions":
                    QMessageBox.warning(self, 
                                        "Invalid Mode for Model",
                                        f"The selected model is a '{model.task}' model. "
                                        "The 'Predictions' feature mode is only available for 'classify' models. "
                                        "Reverting to 'Embed Features' mode.")

                    # Force the feature mode combo box back to "Embed Features"
                    self.model_settings_widget.feature_mode_combo.setCurrentText("Embed Features")
                    
                    # On failure, reset the model cache
                    self.loaded_model = None
                    self.current_feature_generating_model = None
                    return None

                # Update the cache key to the new successful combination
                self.current_feature_generating_model = current_run_key
                self.loaded_model = model
                
                # Get the imgsz, but if it's larger than 128, default to 128
                imgsz = min(getattr(model.model.args, 'imgsz', 128), 128)
                self.imgsz = imgsz
                
                # Warm up the model
                dummy_image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                model.predict(dummy_image, imgsz=imgsz, half=True, device=self.device, verbose=False)
                
                return model
                
            except Exception as e:
                QMessageBox.critical(self, 
                                     "Model Load Error",
                                     f"Could not load the YOLO model '{model_name}'.\n\nError: {e}")
                
                # On failure, reset the model cache
                self.loaded_model = None
                self.current_feature_generating_model = None
                return None
        
        # Model already loaded and cached, return it and its image size
        return self.loaded_model

    def _load_transformer_model(self, model_name):
        """
        Helper function to load a transformer model and cache it.
        
        Args:
            model_name (str): Name of the transformer model to use (e.g., "google/vit-base-patch16-224")
        
        Returns:
            transformers.pipelines.base.Pipeline: The feature extractor pipeline object, or None if loading fails.
        """
        current_run_key = (model_name, "transformer")
        
        # Force a reload if the model path has changed
        if current_run_key != self.current_feature_generating_model or self.loaded_model is None:
            print(f"Model changed. Loading transformer model {model_name}...")
            
            try:
                # Lazy import to avoid unnecessary dependencies
                from transformers import pipeline
                from huggingface_hub import snapshot_download
                
                # Pre-download the model to show progress if it's not cached
                model_path = snapshot_download(repo_id=model_name, 
                                               allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
                
                # Convert device string to appropriate format for transformers pipeline
                if self.device.startswith('cuda'):
                    # Extract device number from 'cuda:0' format for CUDA GPUs
                    device_num = int(self.device.split(':')[-1]) if ':' in self.device else 0
                elif self.device == 'mps':
                    # MPS (Metal Performance Shaders) - Apple's GPU acceleration for macOS
                    device_num = 'mps'
                else:
                    # Default to CPU for any other device string
                    device_num = -1
                
                # Initialize the feature extractor pipeline with local model path
                feature_extractor = pipeline(
                    model=model_path,
                    task="image-feature-extraction",
                    device=device_num,
                )
                try:
                    image_processor = feature_extractor.image_processor
                    if hasattr(image_processor, 'size'):
                        # For older transformers versions
                        self.imgsz = image_processor.size['height']
                    else:
                        # For newer transformers versions
                        self.imgsz = image_processor.crop_size['height']
                        
                except Exception:
                    self.imgsz = 128
                                    
                # Update the cache key to the new successful combination
                self.current_feature_generating_model = current_run_key
                self.loaded_model = feature_extractor
                
                return feature_extractor
                
            except Exception as e:
                QMessageBox.critical(self, 
                                     "Model Load Error",
                                     f"Could not load the transformer model '{model_name}'.\n\nError: {e}")
                
                # On failure, reset the model cache
                self.loaded_model = None
                self.current_feature_generating_model = None
                return None
        
        # Model already loaded and cached, return it and its image size
        return self.loaded_model

    def _prepare_images_from_data_items(self, data_items, progress_bar=None, format='numpy', target_size=None):
        """
        Prepare images from data items for model prediction.
        
        Args:
            data_items (list): List of AnnotationDataItem objects
            progress_bar (ProgressBar, optional): Progress bar for UI updates
            format (str, optional): Output format, either 'numpy' or 'pil'. Default is 'numpy'.
            target_size (tuple, optional): Target size for resizing (width, height). If None, no resizing is performed.
        
        Returns:
            tuple: (image_list, valid_data_items)
        """
        if progress_bar:
            progress_bar.set_title("Preparing images...")
            progress_bar.start_progress(len(data_items))
        
        image_list, valid_data_items = [], []
        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                # Always convert to PIL first for easier resizing
                pil_img = pixmap_to_pil(pixmap)
                
                # Resize if target size is specified
                if target_size and isinstance(target_size, (tuple, list)) and len(target_size) == 2:
                    pil_img = pil_img.resize(target_size, resample=2)  # 2 = PIL.Image.BILINEAR
                
                # Convert to the requested format
                if format.lower() == 'pil':
                    image_list.append(pil_img)
                else:  # Convert to numpy
                    img_array = np.array(pil_img)
                    image_list.append(img_array)
                    
                valid_data_items.append(item)
            
            if progress_bar:
                progress_bar.update_progress()
        
        return image_list, valid_data_items

    def _process_model_results(self, results, valid_data_items, feature_mode, progress_bar=None):
        """
        Process model results and update data item tooltips.
        
        Args:
            results: Model prediction results
            valid_data_items (list): List of valid data items
            feature_mode (str): Mode for feature extraction
            progress_bar (ProgressBar, optional): Progress bar for UI updates
        
        Returns:
            tuple: (features_list, probabilities_dict)
        """
        features_list = []
        probabilities_dict = {}
        
        # Get class names from the model for better tooltips
        model = self.loaded_model.model if hasattr(self.loaded_model, 'model') else None
        class_names = model.names if model and hasattr(model, 'names') else {}
        
        for i, result in enumerate(results):
            if i >= len(valid_data_items):
                break
                
            item = valid_data_items[i]
            ann_id = item.annotation.id
            
            if feature_mode == "Embed Features":
                embedding = result.cpu().numpy().flatten()
                features_list.append(embedding)
                
            elif hasattr(result, 'probs') and result.probs is not None:
                try:
                    probs = result.probs.data.cpu().numpy().squeeze()
                    features_list.append(probs)
                    probabilities_dict[ann_id] = probs

                    # Store the probabilities directly on the data item for confidence sorting
                    item.prediction_probabilities = probs

                    # Format and store prediction details for tooltips
                    # This check will fail with a TypeError if probs is a scalar (unsized)
                    if len(probs) > 0:
                        # Get top 5 predictions
                        top_indices = probs.argsort()[::-1][:5]
                        top_probs = probs[top_indices]

                        formatted_preds = ["<b>Top Predictions:</b>"]
                        for idx, prob in zip(top_indices, top_probs):
                            class_name = class_names.get(int(idx), f"Class {idx}")
                            formatted_preds.append(f"{class_name}: {prob*100:.1f}%")

                        item.prediction_details = "<br>".join(formatted_preds)
                        
                except TypeError:
                    # This error is raised if len(probs) fails on a scalar value.
                    raise TypeError(
                        "The selected model is not compatible with 'Predictions' mode. "
                        "Its output does not appear to be a list of class probabilities. "
                        "Try using 'Embed Features' mode instead."
                    )
            else:
                raise TypeError(
                    "The 'Predictions' feature mode requires a classification model "
                    "(e.g., 'yolov8n-cls.pt') that returns class probabilities. "
                    "The selected model did not provide this output. "
                    "Please use 'Embed Features' mode for this model."
                )
                
            if progress_bar:
                progress_bar.update_progress()
        
        # After processing is complete, update tooltips
        for item in valid_data_items:
            if hasattr(item, 'update_tooltip'):
                item.update_tooltip()
                
        return features_list, probabilities_dict

    def _extract_color_features(self, data_items, progress_bar=None, bins=32):
        """
        Extracts color-based features from annotation crops.

        Features extracted per annotation:
            - Mean, standard deviation, skewness, and kurtosis for each RGB channel
            - Normalized histogram for each RGB channel
            - Grayscale statistics: mean, std, range
            - Geometric features: area, perimeter (if available)
        Returns:
            features: np.ndarray of shape (N, feature_dim)
            valid_data_items: list of AnnotationDataItem with valid crops
        """
        if progress_bar:
            progress_bar.set_title("Extracting features...")
            progress_bar.start_progress(len(data_items))

        features = []
        valid_data_items = []

        for item in data_items:
            pixmap = item.annotation.get_cropped_image()
            if pixmap and not pixmap.isNull():
                # Convert QPixmap to numpy array (H, W, 3)
                arr = pixmap_to_numpy(pixmap)
                pixels = arr.reshape(-1, 3)

                # Basic color statistics
                mean_color = np.mean(pixels, axis=0)
                std_color = np.std(pixels, axis=0)

                # Skewness and kurtosis for each channel
                epsilon = 1e-8  # Prevent division by zero
                centered_pixels = pixels - mean_color
                skew_color = np.mean(centered_pixels ** 3, axis=0) / (std_color ** 3 + epsilon)
                kurt_color = np.mean(centered_pixels ** 4, axis=0) / (std_color ** 4 + epsilon) - 3

                # Normalized histograms for each channel
                histograms = [
                    np.histogram(pixels[:, i], bins=bins, range=(0, 255))[0]
                    for i in range(3)
                ]
                histograms = [
                    h / h.sum() if h.sum() > 0 else np.zeros(bins)
                    for h in histograms
                ]

                # Grayscale statistics
                gray_arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
                grayscale_stats = np.array([
                    np.mean(gray_arr),
                    np.std(gray_arr),
                    np.ptp(gray_arr)
                ])

                # Geometric features (area, perimeter)
                area = getattr(item.annotation, 'area', 0.0)
                perimeter = getattr(item.annotation, 'perimeter', 0.0)
                geometric_features = np.array([area, perimeter])

                # Concatenate all features into a single vector
                current_features = np.concatenate([
                    mean_color,
                    std_color,
                    skew_color,
                    kurt_color,
                    *histograms,
                    grayscale_stats,
                    geometric_features
                ])

                features.append(current_features)
                valid_data_items.append(item)

            if progress_bar:
                progress_bar.update_progress()

        return np.array(features), valid_data_items

    def _extract_yolo_features(self, data_items, model_info, progress_bar=None):
        """Extracts features from annotation crops using a YOLO model."""
        model_name, feature_mode = model_info
        
        # Load the model
        model = self._load_yolo_model(model_name, feature_mode)
        if model is None:
            return np.array([]), []
        
        # Prepare images from data items with proper resizing
        image_list, valid_data_items = self._prepare_images_from_data_items(
            data_items, 
            progress_bar, 
            format='numpy',
            target_size=(self.imgsz, self.imgsz)
        )
        
        if not valid_data_items:
            return np.array([]), []
        
        # Set up prediction parameters
        kwargs = {
            'stream': True,
            'imgsz': self.imgsz,
            'half': True,
            'device': self.device,
            'verbose': False
        }
        
        # Get results based on feature mode
        if feature_mode == "Embed Features":
            results_generator = model.embed(image_list, **kwargs)
        else:
            results_generator = model.predict(image_list, **kwargs)
        
        if progress_bar:
            progress_bar.set_title("Extracting features...")
            progress_bar.start_progress(len(valid_data_items))
        
        try:
            features_list, _ = self._process_model_results(results_generator,
                                                           valid_data_items,
                                                           feature_mode,
                                                           progress_bar=progress_bar)

            return np.array(features_list), valid_data_items

        except TypeError as e:
            QMessageBox.warning(self, "Model Incompatibility Error", str(e))
            return np.array([]), []  # Return empty results to safely stop the pipeline

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_transformer_features(self, data_items, model_name, progress_bar=None):
        """
        Extract features using transformer models from HuggingFace.
        
        Args:
            data_items: List of AnnotationDataItem objects
            model_name: Name of the transformer model to use
            progress_bar: Optional progress bar for tracking
            
        Returns:
            tuple: (features array, valid data items list)
        """
        try:
            if progress_bar:
                progress_bar.set_busy_mode(f"Loading model {model_name}...")
                
            # Load the model with caching support
            feature_extractor = self._load_transformer_model(model_name)
            
            if feature_extractor is None:
                print(f"Failed to load transformer model: {model_name}")
                return np.array([]), []
            
            # Prepare images from data items - get PIL images directly with proper sizing
            image_list, valid_data_items = self._prepare_images_from_data_items(
                data_items, 
                progress_bar, 
                format='pil', 
                target_size=(self.imgsz, self.imgsz)
            )
            
            if not image_list:
                return np.array([]), []
            
            if progress_bar:
                progress_bar.set_title("Extracting features...")
                progress_bar.start_progress(len(valid_data_items))
            
            features_list = []
            valid_items = []
            
            # Process images in batches or individually
            for i, image in enumerate(image_list):
                try:
                    # Extract features
                    features = feature_extractor(image)
                    
                    # Handle different output formats from transformers
                    if isinstance(features, list):
                        feature_tensor = features[0] if len(features) > 0 else features
                    else:
                        feature_tensor = features
                    
                    # Convert to numpy array, handling GPU tensors properly
                    if hasattr(feature_tensor, 'cpu'):
                        # Move tensor to CPU before converting to numpy
                        feature_vector = feature_tensor.cpu().numpy().flatten()
                    else:
                        # Already numpy array or other CPU-compatible format
                        feature_vector = np.array(feature_tensor).flatten()
                    
                    features_list.append(feature_vector)
                    valid_items.append(valid_data_items[i])
                    
                except Exception as e:
                    print(f"Error extracting features for item {i}: {e}")
                    
                finally:
                    if progress_bar:
                        progress_bar.update_progress()

            # Make sure we have consistent feature dimensions
            if features_list:
                features_array = np.array(features_list)
                return features_array, valid_items
            else:
                return np.array([]), []
        
        except Exception as e:
            QMessageBox.warning(self, 
                                "Feature Extraction Error",
                                f"An error occurred during transformer feature extraction.\n\nError: {e}")
            
            return np.array([]), []
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_features(self, data_items, progress_bar=None):
        """Dispatcher to call the appropriate feature extraction function."""
        # Get the selected model and feature mode from the model settings widget
        model_name, feature_mode = self.model_settings_widget.get_selected_model()

        if isinstance(model_name, tuple):
            model_name = model_name[0]

        if not model_name:
            return np.array([]), []

        # Check if it's Color Features first
        if model_name == "Color Features":
            return self._extract_color_features(data_items, progress_bar=progress_bar)

        # Then check if it's a YOLO model (file path with .pt)
        elif is_yolo_model(model_name):
            return self._extract_yolo_features(data_items, (model_name, feature_mode), progress_bar=progress_bar)
        
        # Finally check if it's a transformer model using the shared utility function
        elif is_transformer_model(model_name):
            return self._extract_transformer_features(data_items, model_name, progress_bar=progress_bar)

        return np.array([]), []

    def _run_dimensionality_reduction(self, features, params):
        """
        Runs dimensionality reduction with optional PCA preprocessing.
        For LDA, fits on labeled data and transforms all data.
        """
        technique = params.get('technique', 'UMAP')
        pca_components = params.get('pca_components', 50)
        n_components = params.get('dimensions', 3)
        perform_pca_before = params.get('perform_pca_before', True)

        if len(features) <= n_components:
            return None

        try:
            features_scaled = StandardScaler().fit_transform(features)
            
            # Apply PCA preprocessing if enabled and not PCA technique
            if perform_pca_before and technique != "PCA" and features_scaled.shape[1] > pca_components:
                pca_components = min(pca_components, features_scaled.shape[0] - 1, features_scaled.shape[1])
                print(f"Applying PCA preprocessing to {pca_components} components before {technique}")
                pca = PCA(n_components=pca_components, random_state=42)
                features_scaled = pca.fit_transform(features_scaled)
                variance_explained = sum(pca.explained_variance_ratio_) * 100
                print(f"Variance explained by PCA: {variance_explained:.1f}%")

            if technique == "LDA":
                # Separate labeled and unlabeled data
                labels = []
                labeled_indices = []
                unlabeled_indices = []
                for i, item in enumerate(self.current_data_items):
                    label_name = getattr(item.effective_label, 'short_label_code', REVIEW_LABEL)
                    if label_name != REVIEW_LABEL:
                        labels.append(label_name)
                        labeled_indices.append(i)
                    else:
                        unlabeled_indices.append(i)
                
                if len(set(labels)) < 2:
                    QMessageBox.warning(self, "LDA Error", 
                                        f"LDA requires at least 2 classes (not '{REVIEW_LABEL}').")
                    return None
                
                labeled_features = features_scaled[labeled_indices]
                n_components_lda = min(n_components, len(set(labels)) - 1)
                reducer = LDA(n_components=n_components_lda)
                reducer.fit(labeled_features, labels)
                
                # Transform all data
                return reducer.transform(features_scaled)
            
            # Existing logic for other techniques...
            if technique == "UMAP":
                n_neighbors = min(params.get('n_neighbors', 15), len(features_scaled) - 1)
                reducer = UMAP(
                    n_components=n_components,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=params.get('min_dist', 0.1),
                    metric=params.get('metric', 'cosine')
                )
            elif technique == "TSNE":
                perplexity = min(params.get('perplexity', 30), len(features_scaled) - 1)
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=perplexity,
                    early_exaggeration=params.get('early_exaggeration', 12.0),
                    learning_rate=params.get('learning_rate', 'auto'),
                    init='pca'
                )
            elif technique == "PCA":
                reducer = PCA(n_components=n_components, random_state=42)
            else:
                return None

            return reducer.fit_transform(features_scaled)

        except Exception as e:
            QMessageBox.warning(self, 
                                "Embedding Error",
                                f"An error occurred during dimensionality reduction with {technique}.\n\nError: {e}")
            
            return None

    def _update_data_items_with_embedding(self, data_items, embedded_features):
        """Updates AnnotationDataItem objects with embedding results for 2D or 3D data."""
        if embedded_features is None:
            print("Error: No embedded features to process.")
            return

        n_dims = embedded_features.shape[1]
        if n_dims not in [2, 3]:
            print(f"Error: Expected 2D or 3D embedded features, but got {n_dims}D.")
            return

        scale_factor = 4000
        min_vals = np.min(embedded_features, axis=0)
        max_vals = np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals
        
        # Avoid division by zero if a dimension has no variance
        range_vals[range_vals == 0] = 1

        for i, item in enumerate(data_items):
            # Normalize each dimension to a range of [0, 1]
            norm_coords = (embedded_features[i] - min_vals) / range_vals
            
            # Scale to the scene size and center around (0,0,0)
            scaled_coords = (norm_coords * scale_factor) - (scale_factor / 2)
            
            if n_dims == 3:
                # Store the original, un-rotated 3D coordinates
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = scaled_coords[2]
            else:  # n_dims == 2
                # Store the 2D coordinates and set Z to 0 for a flat plot
                item.embedding_x_3d = scaled_coords[0]
                item.embedding_y_3d = scaled_coords[1]
                item.embedding_z_3d = 0.0

            # Set the initial projected coordinates (no rotation)
            item.embedding_x = item.embedding_x_3d
            item.embedding_y = item.embedding_y_3d
            item.embedding_z = item.embedding_z_3d  # z represents depth

            item.embedding_id = i

    def run_embedding_pipeline(self):
        """
        Orchestrates feature extraction and dimensionality reduction.
        If the EmbeddingViewer is in isolate mode, it will use only the visible
        (isolated) points as input for the pipeline.
        """
        items_to_embed = []
        if self.embedding_viewer.isolated_mode:
            items_to_embed = [point.data_item for point in self.embedding_viewer.isolated_points]
        else:
            items_to_embed = self.current_data_items

        if not items_to_embed:
            print("No items to process for embedding.")
            return

        self.annotation_viewer.clear_selection()
        if self.annotation_viewer.isolated_mode:
            self.annotation_viewer.show_all_annotations()

        self.embedding_viewer.render_selection_from_ids(set())
        
        # Reset sort to "None" to ensure confidence colors/bins recalculate correctly
        self.annotation_viewer.sort_combo.setCurrentText("None")
        
        self.update_button_states()

        self.current_embedding_model_info = self.model_settings_widget.get_selected_model()

        embedding_params = self.embedding_settings_widget.get_embedding_parameters()
        selected_model, selected_feature_mode = self.current_embedding_model_info

        # If the model name is a path, use only its base name.
        if os.path.sep in selected_model or '/' in selected_model:
            sanitized_model_name = os.path.basename(selected_model)
        else:
            sanitized_model_name = selected_model

        # Replace characters that might be problematic in filenames
        sanitized_model_name = sanitized_model_name.replace(' ', '_')
        # Also replace the forward slash to handle "N/A"
        sanitized_feature_mode = selected_feature_mode.replace(' ', '_').replace('/', '_')

        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        progress_bar = ProgressBar(self, "Processing Annotations")
        progress_bar.show()

        try:
            progress_bar.set_busy_mode("Checking feature cache...")
            cached_features, items_to_process = self.feature_store.get_features(items_to_embed, model_key)
            print(f"Found {len(cached_features)} features in cache. Need to compute {len(items_to_process)}.")

            if items_to_process:
                newly_extracted_features, valid_items_processed = self._extract_features(items_to_process,
                                                                                         progress_bar=progress_bar)
                if len(newly_extracted_features) > 0:
                    progress_bar.set_busy_mode("Saving new features to cache...")
                    self.feature_store.add_features(valid_items_processed, newly_extracted_features, model_key)
                    new_features_dict = {item.annotation.id: vec for item, vec in zip(valid_items_processed,
                                                                                      newly_extracted_features)}
                    cached_features.update(new_features_dict)

            if not cached_features:
                print("No features found or computed. Aborting.")
                return

            final_feature_list = []
            final_data_items = []
            for item in items_to_embed:
                if item.annotation.id in cached_features:
                    final_feature_list.append(cached_features[item.annotation.id])
                    final_data_items.append(item)

            features = np.array(final_feature_list)
            self.current_data_items = final_data_items
            self.annotation_viewer.update_annotations(self.current_data_items)
            
            # Store features and model key for wizard access
            self.current_features = features
            self.current_feature_generating_model = model_key

            # Validate LDA requirements if selected
            if embedding_params.get('technique') == 'LDA':
                labeled_count = sum(1 for item in final_data_items 
                                    if getattr(item.effective_label, 'short_label_code', REVIEW_LABEL) != REVIEW_LABEL)
                if labeled_count < 2:
                    QMessageBox.warning(self, "LDA Error", 
                                        f"LDA requires at least 2 labeled annotations (not '{REVIEW_LABEL}').")
                    return

            progress_bar.set_busy_mode("Running dimensionality reduction...")
            embedded_features = self._run_dimensionality_reduction(features, embedding_params)

            if embedded_features is None:
                return

            progress_bar.set_busy_mode("Updating visualization...")
            self._update_data_items_with_embedding(self.current_data_items, embedded_features)
            self.embedding_viewer.update_embeddings(self.current_data_items, embedded_features.shape[1])
            self.embedding_viewer.show_embedding()
            self.embedding_viewer.fit_view_to_points()

            # Check if confidence scores are available to enable sorting
            _, feature_mode = self.current_embedding_model_info
            is_predict_mode = feature_mode == "Predictions"
            if is_predict_mode:
                self.annotation_viewer.set_confidence_sort_availability(True)

            # If using Predictions mode, update data items with probabilities for confidence sorting
            if is_predict_mode:
                for item in self.current_data_items:
                    if item.annotation.id in cached_features:
                        item.prediction_probabilities = cached_features[item.annotation.id]

            # When a new embedding is run, any previous similarity sort becomes irrelevant
            self.annotation_viewer.active_ordered_ids = []
            
            # Auto-calculate anomaly scores and quality metrics
            progress_bar.set_busy_mode("Calculating quality metrics...")
            self._calculate_quality_metrics_silently()
            
            # Enable auto-annotation wizard button now that features are available
            self._update_wizard_button_state()

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()
    
    def _calculate_quality_metrics_silently(self):
        """
        Calculates anomaly scores, local density, spatial consistency, and quality scores
        for all current data items without displaying results or changing selection.
        Similar to find_anomalies but doesn't select/display results.
        """
        # Get parameters (use defaults)
        contamination = self.anomaly_params.get('contamination', 0.1)
        n_neighbors = min(self.anomaly_params.get('n_neighbors', 20), len(self.current_data_items) - 1)
        
        if len(self.current_data_items) < n_neighbors:
            print(f"Not enough data items ({len(self.current_data_items)}) for quality calculation (needs {n_neighbors}).")
            return

        # Get the model key used for the current embedding
        model_name, feature_mode = self.current_embedding_model_info
        sanitized_model_name = os.path.basename(model_name).replace(' ', '_')
        sanitized_feature_mode = feature_mode.replace(' ', '_').replace('/', '_')
        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        try:
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import NearestNeighbors
            
            # Get the high-dimensional features
            features_dict, _ = self.feature_store.get_features(self.current_data_items, model_key)
            if not features_dict:
                print("Could not retrieve features for quality calculation.")
                return

            query_ann_ids = list(features_dict.keys())
            query_vectors = np.array([features_dict[ann_id] for ann_id in query_ann_ids]).astype('float32')

            # 1. Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
            lof_predictions = lof.fit_predict(query_vectors)
            lof_scores = -lof.negative_outlier_factor_
            
            # 2. Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_predictions = iso_forest.fit_predict(query_vectors)
            iso_scores = -iso_forest.score_samples(query_vectors)
            
            # Normalize scores
            lof_scores_normalized = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
            iso_scores_normalized = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)
            
            # Combine scores
            combined_scores = 0.6 * lof_scores_normalized + 0.4 * iso_scores_normalized
            
            # Calculate local density
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
            nn.fit(query_vectors)
            distances, indices = nn.kneighbors(query_vectors)
            
            avg_distances = distances[:, 1:].mean(axis=1)
            local_densities = 1.0 / (avg_distances + 1e-10)
            
            # Calculate spatial consistency
            spatial_consistencies = []
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                label_id = data_item.effective_label.id
                
                neighbor_indices = indices[i][1:]
                matching_neighbors = 0
                valid_neighbors = 0
                
                for neighbor_idx in neighbor_indices:
                    neighbor_ann_id = query_ann_ids[neighbor_idx]
                    neighbor_item = self.data_item_cache[neighbor_ann_id]
                    
                    if neighbor_item.annotation.image_path == data_item.annotation.image_path:
                        valid_neighbors += 1
                        if neighbor_item.effective_label.id == label_id:
                            matching_neighbors += 1
                
                consistency = matching_neighbors / valid_neighbors if valid_neighbors > 0 else 0.5
                spatial_consistencies.append(consistency)
            
            spatial_consistencies = np.array(spatial_consistencies)
            
            # Update data items with scores
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                
                # Store metrics
                data_item.anomaly_score = float(combined_scores[i])
                data_item.local_density = float(local_densities[i])
                data_item.spatial_consistency = float(spatial_consistencies[i])
                
                # Calculate quality score
                data_item.calculate_quality_score()
                
                # Update tooltip
                if hasattr(data_item, 'point_item') and data_item.point_item:
                    data_item.point_item.update_tooltip()
            
            print(f"Quality metrics calculated for {len(query_ann_ids)} annotations.")

        except ImportError as e:
            print(f"Could not calculate quality metrics: {str(e)}")
        except Exception as e:
            print(f"Error during quality calculation: {str(e)}")

    def refresh_filters(self):
        """Refresh display: filter data and update annotation viewer."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.current_data_items = self.get_filtered_data_items()
            self.current_features = None
            self.annotation_viewer.update_annotations(self.current_data_items)
            self.embedding_viewer.clear_points()
            self.embedding_viewer.show_placeholder()

            # Reset sort options when filters change
            self.annotation_viewer.active_ordered_ids = []
            
            # Update the annotation count in the label window
            self.label_window.update_annotation_count()

        finally:
            QApplication.restoreOverrideCursor()

    def on_label_selected_for_preview(self, label):
        """Handle label selection to update preview state."""
        if hasattr(self, 'annotation_viewer') and self.annotation_viewer.selected_widgets:
            self.annotation_viewer.apply_preview_label_to_selected(label)
            self.update_button_states()

    def delete_data_items(self, data_items_to_delete):
        """
        Permanently deletes a list of data items and their associated annotations
        and visual components from the explorer and the main application.
        """
        if not data_items_to_delete:
            return

        print(f"Permanently deleting {len(data_items_to_delete)} item(s).")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            deleted_ann_ids = {item.annotation.id for item in data_items_to_delete}
            annotations_to_delete_from_main_app = [item.annotation for item in data_items_to_delete]

            # 1. Delete from the main application's data store
            self.annotation_window.delete_annotations(annotations_to_delete_from_main_app)

            # 2. Remove from Explorer's internal data structures
            self.current_data_items = [
                item for item in self.current_data_items if item.annotation.id not in deleted_ann_ids
            ]
            # Also update the annotation viewer's list to keep it in sync
            self.annotation_viewer.all_data_items = [
                item for item in self.annotation_viewer.all_data_items if item.annotation.id not in deleted_ann_ids
            ]
            for ann_id in deleted_ann_ids:
                if ann_id in self.data_item_cache:
                    del self.data_item_cache[ann_id]

            # 3. Remove from AnnotationViewer
            blocker = QSignalBlocker(self.annotation_viewer)  # Block signals during mass removal
            for ann_id in deleted_ann_ids:
                if ann_id in self.annotation_viewer.annotation_widgets_by_id:
                    widget = self.annotation_viewer.annotation_widgets_by_id.pop(ann_id)
                    if widget in self.annotation_viewer.selected_widgets:
                        self.annotation_viewer.selected_widgets.remove(widget)
                    widget.setParent(None)
                    widget.deleteLater()
            blocker.unblock()
            self.annotation_viewer.recalculate_layout()

            # 4. Remove from EmbeddingViewer
            blocker = QSignalBlocker(self.embedding_viewer.graphics_scene)
            for ann_id in deleted_ann_ids:
                if ann_id in self.embedding_viewer.points_by_id:
                    point = self.embedding_viewer.points_by_id.pop(ann_id)
                    self.embedding_viewer.graphics_scene.removeItem(point)
            blocker.unblock()
            self.embedding_viewer.on_selection_changed()  # Trigger update of selection state

            # 5. Update UI
            self.update_label_window_selection()
            self.update_button_states()

            # 6. Refresh main window annotations table counts
            # Note: We don't call load_annotations() here to avoid bringing MainWindow forward.
            # The update_image_annotations() call updates the table counts, which is sufficient.
            affected_images = {ann.image_path for ann in annotations_to_delete_from_main_app}
            for image_path in affected_images:
                self.image_window.update_image_annotations(image_path)

        except Exception as e:
            QMessageBox.warning(self, 
                                "Deletion Error",
                                f"An error occurred while deleting annotations.\n\nError: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def clear_preview_changes(self):
        """
        Clears all preview changes in the annotation viewer, reverts tooltips,
        and clears any active selections.
        """
        # First, clear any active selections from the UI.
        self._clear_selections()
        
        if hasattr(self, 'annotation_viewer'):
            self.annotation_viewer.clear_preview_states()

            # After reverting, tooltips need to be updated to reflect original labels
            for widget in self.annotation_viewer.annotation_widgets_by_id.values():
                widget.update_tooltip()
            for point in self.embedding_viewer.points_by_id.values():
                point.update_tooltip()

        # After reverting all changes, update the label and count display
        self.update_label_window_selection()
        
        # After reverting all changes, update the button states.
        self.update_button_states()
        print("Cleared all pending changes.")

    def update_button_states(self):
        """Update the state of Clear Preview, Apply, and Find Similar buttons."""
        has_changes = self.annotation_viewer.has_preview_changes()
        self.clear_preview_button.setEnabled(has_changes)
        self.apply_button.setEnabled(has_changes)
        
        # Update tooltips with a summary of changes
        summary = self.annotation_viewer.get_preview_changes_summary()
        self.clear_preview_button.setToolTip(f"Clear all preview changes - {summary}")
        self.apply_button.setToolTip(f"Apply changes - {summary}")

        # Find Similar button is now managed by embedding_viewer's _update_toolbar_state()
        # No need to manually update it here since it updates automatically on selection changes

    def apply(self):
        """
        Apply all pending label modifications to the main application's data.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # --- 1. Process Label Changes ---
            applied_label_changes = []
            # Iterate over all current data items
            for item in self.current_data_items:
                if item.apply_preview_permanently():
                    applied_label_changes.append(item.annotation)

            # --- 2. Update UI if any changes were made ---
            if not applied_label_changes:
                print("No pending changes to apply.")
                return

            # Update the main application's data and UI
            # Note: We don't call load_annotations() here to avoid bringing MainWindow forward
            # and because it's unnecessary - the Explorer manages its own view updates.
            affected_images = {ann.image_path for ann in applied_label_changes}
            for image_path in affected_images:
                self.image_window.update_image_annotations(image_path)

            # Refresh the annotation viewer since its underlying data has changed.
            # This implicitly deselects everything by rebuilding the widgets.
            self.annotation_viewer.update_annotations(self.current_data_items)

            # Explicitly clear selections and update UI states for consistency.
            self._clear_selections()

            print("Applied changes successfully.")

        except Exception as e:
            QMessageBox.warning(self, 
                                "Apply Error",
                                f"An error occurred while applying changes.\n\nError: {e}")
        finally:
            QApplication.restoreOverrideCursor()
    
    # ============================================================================
    # ML Training and Prediction Methods
    # ============================================================================
    
    def get_features_for_training(self, data_items, feature_type='full'):
        """
        Extract features for model training from data items.
        
        Args:
            data_items: List of AnnotationDataItem objects
            feature_type: 'full' for high-dimensional features, 'reduced' for embeddings
            
        Returns:
            numpy array of features (n_samples, n_features)
        """
        if feature_type == 'reduced':
            # Use reduced embeddings (2D or 3D)
            features = []
            for item in data_items:
                if hasattr(item, 'embedding_x') and item.embedding_x is not None:
                    feat = [item.embedding_x, item.embedding_y]
                    if hasattr(item, 'embedding_z') and item.embedding_z is not None:
                        feat.append(item.embedding_z)
                    features.append(feat)
                else:
                    # No embeddings available
                    return None
            
            return np.array(features)
        
        else:  # 'full'
            # Use full features from FeatureStore
            if not self.current_feature_generating_model:
                raise AutoAnnotationError("No features available. Please run embedding pipeline first.")
            
            model_key = self.current_feature_generating_model
            found_features, not_found = self.feature_store.get_features(data_items, model_key)
            
            if not_found:
                # Extract features for missing items
                progress_bar = ProgressBar(self, title="Extracting Features")
                try:
                    self._extract_features(not_found, progress_bar)
                    found_features, still_missing = self.feature_store.get_features(data_items, model_key)
                    
                    if still_missing:
                        raise AutoAnnotationError(f"Failed to extract features for {len(still_missing)} items")
                finally:
                    progress_bar.close()
            
            # Build feature matrix in correct order
            features = []
            for item in data_items:
                feat = found_features.get(item.annotation.id)
                if feat is not None:
                    features.append(feat)
                else:
                    raise AutoAnnotationError(f"Missing features for annotation {item.annotation.id}")
            
            return np.array(features)
    
    def train_annotation_model(self, feature_type='full', model_type='random_forest', model_params=None):
        """
        Train a scikit-learn model on labeled annotations.
        
        Args:
            feature_type: 'full' or 'reduced'
            model_type: 'random_forest', 'svc', or 'knn'
            model_params: Dictionary of model-specific parameters
            
        Returns:
            Dictionary containing model, scaler, classes, and metrics
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        except ImportError:
            raise AutoAnnotationError("scikit-learn not installed. Please install: pip install scikit-learn")
        
        # Filter data items - only use labeled (non-Review) annotations
        labeled_items = [
            item for item in self.current_data_items
            if getattr(item.effective_label, 'short_label_code', '') != REVIEW_LABEL
        ]
        
        if len(labeled_items) < 2:
            raise AutoAnnotationError("Need at least 2 labeled annotations to train model")
        
        # Get unique classes
        label_to_items = {}
        for item in labeled_items:
            label_code = item.effective_label.short_label_code
            if label_code not in label_to_items:
                label_to_items[label_code] = []
            label_to_items[label_code].append(item)
        
        classes = sorted(label_to_items.keys())
        
        if len(classes) < 2:
            raise AutoAnnotationError("Need at least 2 different classes to train model")
        
        # Check for cold start / class imbalance
        min_samples = min(len(items) for items in label_to_items.values())
        max_samples = max(len(items) for items in label_to_items.values())
        
        if min_samples < 2:
            raise AutoAnnotationError(f"Some classes have fewer than 2 examples. Cannot train.")
        
        warnings_text = []
        if min_samples < 5:
            warnings_text.append(f"⚠️ Cold start detected: minimum {min_samples} examples per class")
        
        if max_samples / min_samples > 5:
            warnings_text.append(f"⚠️ Class imbalance: {max_samples}:{min_samples} ratio")
        
        # Extract features
        try:
            X = self.get_features_for_training(labeled_items, feature_type)
            if X is None:
                raise AutoAnnotationError(f"No {feature_type} features available")
        except Exception as e:
            raise AutoAnnotationError(f"Failed to extract features: {str(e)}")
        
        # Create labels
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        y = np.array([class_to_idx[item.effective_label.short_label_code] 
                     for item in labeled_items])
        
        # Normalize features (especially important for SVC)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create model
        if model_params is None:
            model_params = {}
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=42,
                class_weight='balanced'  # Handle imbalance
            )
        elif model_type == 'svc':
            model = SVC(
                C=model_params.get('C', 1.0),
                kernel=model_params.get('kernel', 'rbf'),
                probability=True,  # Enable probability estimates
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=min(model_params.get('n_neighbors', 5), len(labeled_items) - 1)
            )
        else:
            raise AutoAnnotationError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        # Get classification report
        report = classification_report(y, y_pred, target_names=classes, zero_division=0)
        
        # Get confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        result = {
            'model': model,
            'scaler': scaler,
            'classes': classes,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'warnings': warnings_text,
            'n_samples': len(labeled_items),
            'feature_type': feature_type
        }
        
        return result
    
    def predict_with_model(self, data_items, model, scaler, class_to_idx, idx_to_class, feature_type='full'):
        """
        Predict labels for data items using trained model.
        
        Args:
            data_items: List of AnnotationDataItem objects
            model: Trained scikit-learn model
            scaler: Fitted StandardScaler
            class_to_idx: Dict mapping class names to indices
            idx_to_class: Dict mapping indices to class names
            feature_type: 'full' or 'reduced'
            
        Returns:
            None (predictions stored in data_items.ml_prediction)
        """
        if not data_items:
            return
        
        # Extract features
        try:
            X = self.get_features_for_training(data_items, feature_type)
            if X is None:
                return
        except Exception as e:
            print(f"Failed to extract features for prediction: {e}")
            return
        
        # Normalize
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        probabilities = model.predict_proba(X_scaled)
        
        # Store predictions in data items
        for i, item in enumerate(data_items):
            probs = probabilities[i]
            
            # Get top prediction
            top_idx = np.argmax(probs)
            top_label = idx_to_class[top_idx]
            top_conf = probs[top_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(probs)[-3:][::-1]
            top_predictions = [
                {'label': idx_to_class[idx], 'confidence': probs[idx]}
                for idx in top_3_indices
            ]
            
            # Calculate margin (difference between top 2)
            if len(probs) > 1:
                sorted_probs = np.sort(probs)
                margin = sorted_probs[-1] - sorted_probs[-2]
            else:
                margin = 1.0
            
            item.ml_prediction = {
                'label': top_label,
                'confidence': float(top_conf),
                'margin': float(margin),
                'top_predictions': top_predictions,
                'probabilities': {idx_to_class[j]: float(probs[j]) 
                                for j in range(len(probs))}
            }
    
    def get_next_annotation_batch(self, model, scaler, class_to_idx, feature_type='full', batch_size=20):
        """
        Get next batch of annotations to label using active learning strategies.
        Combines uncertainty sampling, margin sampling, and diversity.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            class_to_idx: Class mapping
            feature_type: Feature type used
            batch_size: Number of annotations to return
            
        Returns:
            List of AnnotationDataItem objects sorted by priority
        """
        # Get unlabeled (Review) annotations
        unlabeled_items = [
            item for item in self.current_data_items
            if getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        ]
        
        if not unlabeled_items:
            return []
        
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        # Get predictions for all unlabeled items
        self.predict_with_model(unlabeled_items, model, scaler, class_to_idx, idx_to_class, feature_type)
        
        # Score items for active learning
        scored_items = []
        for item in unlabeled_items:
            if not hasattr(item, 'ml_prediction'):
                continue
            
            pred = item.ml_prediction
            
            # Uncertainty score (low confidence = high priority)
            uncertainty_score = 1.0 - pred['confidence']
            
            # Margin score (small margin = high priority)
            margin_score = 1.0 - pred['margin']
            
            # Diversity score (distance from training set in feature space)
            # For now, use a simple heuristic based on prediction entropy
            probs = list(pred['probabilities'].values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
            
            # Combined score (weighted average)
            combined_score = (0.4 * uncertainty_score + 
                            0.4 * margin_score + 
                            0.2 * diversity_score)
            
            scored_items.append((combined_score, item))
        
        # Sort by score (highest first) and return top batch_size
        scored_items.sort(key=lambda x: x[0], reverse=True)
        batch = [item for score, item in scored_items[:batch_size]]
        
        return batch
    
    def auto_label_confident_predictions(self, model, scaler, class_to_idx, idx_to_class, 
                                        feature_type='full', threshold=0.95):
        """
        Automatically apply labels to high-confidence predictions.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            class_to_idx: Class mapping
            idx_to_class: Reverse class mapping
            feature_type: Feature type
            threshold: Minimum confidence for auto-labeling
            
        Returns:
            Number of annotations auto-labeled
        """
        # Get Review annotations
        review_items = [
            item for item in self.current_data_items
            if getattr(item.effective_label, 'short_label_code', '') == REVIEW_LABEL
        ]
        
        if not review_items:
            return 0
        
        # Get predictions
        self.predict_with_model(review_items, model, scaler, class_to_idx, idx_to_class, feature_type)
        
        # Apply labels where confidence > threshold
        count = 0
        for item in review_items:
            if not hasattr(item, 'ml_prediction'):
                continue
            
            pred = item.ml_prediction
            if pred['confidence'] >= threshold:
                # Find label object
                predicted_label_code = pred['label']
                label_obj = None
                for label in self.main_window.label_window.labels:
                    if label.short_label_code == predicted_label_code:
                        label_obj = label
                        break
                
                if label_obj:
                    item.annotation.update_label(label_obj)
                    count += 1
        
        # Update displays
        if count > 0:
            self.annotation_viewer.update_annotations(self.current_data_items)
            self.embedding_viewer.update_embeddings(self.current_data_items, 
                                                   2 if not hasattr(self.current_data_items[0], 'embedding_z') else 3)
        
        return count
    
    def validate_training_labels(self):
        """
        Use anomaly detection to identify potentially mislabeled training data.
        
        Returns:
            List of AnnotationDataItem objects that may be mislabeled
        """
        # Get labeled (non-Review) annotations
        labeled_items = [
            item for item in self.current_data_items
            if getattr(item.effective_label, 'short_label_code', '') != REVIEW_LABEL
        ]
        
        if len(labeled_items) < 10:
            return []
        
        # Run anomaly detection on labeled set
        # Group by label and find anomalies within each group
        from collections import defaultdict
        label_groups = defaultdict(list)
        
        for item in labeled_items:
            label_code = item.effective_label.short_label_code
            label_groups[label_code].append(item)
        
        flagged_items = []
        
        for label_code, items in label_groups.items():
            if len(items) < 5:  # Skip small groups
                continue
            
            # Run anomaly detection on this group
            # Temporarily set these as the data items
            original_items = self.current_data_items
            self.current_data_items = items
            
            try:
                # Use existing anomaly detection
                self.find_anomalies()
                
                # Items with high anomaly scores are potentially mislabeled
                for item in items:
                    if hasattr(item, 'anomaly_score') and item.anomaly_score > 0.7:
                        flagged_items.append(item)
            finally:
                self.current_data_items = original_items
        
        return flagged_items
    
    def open_auto_annotation_wizard(self):
        """Open the Auto-Annotation Wizard."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependencies",
                "scikit-learn is required for the Auto-Annotation Wizard.\n\n"
                "Install it with: pip install scikit-learn"
            )
            return
        
        # Create wizard if it doesn't exist
        if self.auto_annotation_wizard is None:
            self.auto_annotation_wizard = AutoAnnotationWizard(self, self)
            self.auto_annotation_wizard.annotations_updated.connect(self._on_wizard_annotations_updated)
        
        # Show wizard (modeless)
        self.auto_annotation_wizard.show()
        self.auto_annotation_wizard.raise_()
        self.auto_annotation_wizard.activateWindow()
    
    def _on_wizard_annotations_updated(self, updated_items):
        """Handle annotations updated from wizard."""
        # Refresh views
        self.annotation_viewer.update_annotations(self.current_data_items)
        
        n_dims = 2
        if self.current_data_items and hasattr(self.current_data_items[0], 'embedding_z'):
            if self.current_data_items[0].embedding_z is not None:
                n_dims = 3
        
        self.embedding_viewer.update_embeddings(self.current_data_items, n_dims)
        
        # Update button states
        self.update_button_states()
    
    def _update_wizard_button_state(self):
        """Update the Auto-Annotation Wizard button state."""
        # Enable if we have features or embeddings
        has_features = bool(self.current_feature_generating_model)
        has_embeddings = (self.current_data_items and 
                         hasattr(self.current_data_items[0], 'embedding_x') and
                         self.current_data_items[0].embedding_x is not None)
        
        self.auto_annotation_button.setEnabled(has_features or has_embeddings)
            
    def _cleanup_resources(self):
        """Clean up resources."""
        # Clear any cached normalized indexes
        if hasattr(self, '_normalized_indexes'):
            self._normalized_indexes.clear()
        
        self.loaded_model = None
        self.model_path = ""
        self.current_features = None
        self.current_feature_generating_model = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
