import warnings

import os

import numpy as np
import torch

from ultralytics import YOLO

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSlot
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
                             QMainWindow, QSplitter, QGroupBox, QMessageBox,
                             QApplication)

from coralnet_toolbox.Explorer.QtViewers import AnnotationViewer
from coralnet_toolbox.Explorer.QtViewers import EmbeddingViewer
from coralnet_toolbox.Explorer.QtFeatureStore import FeatureStore
from coralnet_toolbox.Explorer.QtDataItem import AnnotationDataItem
from coralnet_toolbox.Explorer.QtSettingsWidgets import ModelSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import EmbeddingSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import AnnotationSettingsWidget

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
    from umap import UMAP
except ImportError:
    print("Warning: sklearn or umap not installed. Some features may be unavailable.")
    StandardScaler = None
    PCA = None
    TSNE = None
    UMAP = None


warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_WIDTH = 3

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

        self.device = main_window.device
        self.loaded_model = None
        self.imgsz = 128

        self.feature_store = FeatureStore()
        
        # Add a property to store the parameters with defaults
        self.mislabel_params = {'k': 20, 'threshold': 0.6}
        self.uncertainty_params = {'confidence': 0.6, 'margin': 0.1}
        self.similarity_params = {'k': 30}
        self.duplicate_params = {'threshold': 0.05}
        
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
        self.left_layout = QVBoxLayout(self.left_panel)

        self.annotation_settings_widget = None
        self.model_settings_widget = None
        self.embedding_settings_widget = None
        self.annotation_viewer = None
        self.embedding_viewer = None

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

        # Horizontal layout for the three settings panels (original horizontal layout)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.annotation_settings_widget, 2)
        top_layout.addWidget(self.model_settings_widget, 1)
        top_layout.addWidget(self.embedding_settings_widget, 1)
        top_container = QWidget()
        top_container.setLayout(top_layout)

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

        # Create a VERTICAL splitter to manage the height between the settings and viewers.
        # This makes the top settings panel vertically resizable.
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_container)
        main_splitter.addWidget(middle_splitter)
        
        # Set initial heights to give the settings panel a bit more space by default
        main_splitter.setSizes([250, 750]) 

        # Add the new main splitter to the layout instead of the individual components
        self.main_layout.addWidget(main_splitter, 1)

        self.main_layout.addWidget(self.label_window)

        self.buttons_layout = QHBoxLayout()
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
        self.embedding_viewer.find_mislabels_requested.connect(self.find_potential_mislabels)
        self.embedding_viewer.mislabel_parameters_changed.connect(self.on_mislabel_params_changed)
        self.model_settings_widget.selection_changed.connect(self.on_model_selection_changed)
        self.embedding_viewer.find_uncertain_requested.connect(self.find_uncertain_annotations)
        self.embedding_viewer.uncertainty_parameters_changed.connect(self.on_uncertainty_params_changed)
        self.embedding_viewer.find_duplicates_requested.connect(self.find_duplicate_annotations)
        self.embedding_viewer.duplicate_parameters_changed.connect(self.on_duplicate_params_changed)
        self.annotation_viewer.find_similar_requested.connect(self.find_similar_annotations)
        self.annotation_viewer.similarity_settings_widget.parameters_changed.connect(self.on_similarity_params_changed)
        
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
    def on_mislabel_params_changed(self, params):
        """Updates the stored parameters for mislabel detection."""
        self.mislabel_params = params
        print(f"Mislabel detection parameters updated: {self.mislabel_params}")
        
    @pyqtSlot(dict)
    def on_uncertainty_params_changed(self, params):
        """Updates the stored parameters for uncertainty analysis."""
        self.uncertainty_params = params
        print(f"Uncertainty parameters updated: {self.uncertainty_params}")

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
        is_predict_mode = ".pt" in model_name and feature_mode == "Predictions"
        
        self.embedding_viewer.is_uncertainty_analysis_available = is_predict_mode
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
            self.label_window.set_active_label(first_effective_label)
            # This emit is what updates other UI elements, like the annotation list
            self.annotation_window.labelSelected.emit(first_effective_label.id)
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
    
    def find_potential_mislabels(self):
        """
        Identifies annotations whose label does not match the majority of its
        k-nearest neighbors in the high-dimensional feature space.
        Skips any annotation or neighbor with an invalid label (id == -1).
        """
        # Get parameters from the stored property instead of hardcoding
        K = self.mislabel_params.get('k', 5)
        agreement_threshold = self.mislabel_params.get('threshold', 0.6)

        if not self.embedding_viewer.points_by_id or len(self.embedding_viewer.points_by_id) < K:
            QMessageBox.information(self, 
                                    "Not Enough Data",
                                    f"This feature requires at least {K} points in the embedding viewer.")
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
            # Get the FAISS index and the mapping from index to annotation ID
            index = self.feature_store._get_or_load_index(model_key)
            faiss_idx_to_ann_id = self.feature_store.get_faiss_index_to_annotation_id_map(model_key)
            if index is None or not faiss_idx_to_ann_id:
                QMessageBox.warning(self, 
                                    "Error", 
                                    "Could not find a valid feature index for the current model.")
                return

            # Get the high-dimensional features for the points in the current view
            features_dict, _ = self.feature_store.get_features(data_items_in_view, model_key)
            if not features_dict:
                QMessageBox.warning(self, 
                                    "Error", 
                                    "Could not retrieve features for the items in view.")
                return

            query_ann_ids = list(features_dict.keys())
            query_vectors = np.array([features_dict[ann_id] for ann_id in query_ann_ids]).astype('float32')

            # Perform k-NN search. We search for K+1 because the point itself will be the first result.
            _, I = index.search(query_vectors, K + 1)

            mislabeled_ann_ids = []
            for i, ann_id in enumerate(query_ann_ids):
                data_item = self.data_item_cache[ann_id]
                # Use preview_label if present, else effective_label
                label_obj = getattr(data_item, "preview_label", None) or data_item.effective_label
                current_label_id = getattr(label_obj, "id", "-1")
                if current_label_id == "-1":
                    continue  # Skip if label is invalid

                # Get neighbor labels, ignoring the first result (the point itself)
                neighbor_faiss_indices = I[i][1:]

                neighbor_labels = []
                for n_idx in neighbor_faiss_indices:
                    if n_idx in faiss_idx_to_ann_id:
                        neighbor_ann_id = faiss_idx_to_ann_id[n_idx]
                        if neighbor_ann_id in self.data_item_cache:
                            neighbor_item = self.data_item_cache[neighbor_ann_id]
                            neighbor_label_obj = getattr(neighbor_item, "preview_label", None)
                            if neighbor_label_obj is None:
                                neighbor_label_obj = neighbor_item.effective_label
                            neighbor_label_id = getattr(neighbor_label_obj, "id", "-1")
                            if neighbor_label_id != "-1":
                                neighbor_labels.append(neighbor_label_id)

                if not neighbor_labels:
                    continue

                num_matching_neighbors = neighbor_labels.count(current_label_id)
                agreement_ratio = num_matching_neighbors / len(neighbor_labels)

                if agreement_ratio < agreement_threshold:
                    mislabeled_ann_ids.append(ann_id)

            self.embedding_viewer.render_selection_from_ids(set(mislabeled_ann_ids))

        finally:
            QApplication.restoreOverrideCursor()

    def find_duplicate_annotations(self):
        """
        Identifies annotations that are likely duplicates based on feature similarity.
        It uses a nearest-neighbor approach in the high-dimensional feature space.
        For each group of duplicates found, it selects all but one "original".
        """
        threshold = self.duplicate_params.get('threshold', 0.05)

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

        # Make cursor busy
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

            # Find the 2 nearest neighbors for each vector. D = squared L2 distances.
            D, I = index.search(query_vectors, 2)

            # Use a Disjoint Set Union (DSU) data structure to group duplicates.
            parent = {ann_id: ann_id for ann_id in query_ann_ids}
            
            # Helper functions for DSU
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

            for i, ann_id in enumerate(query_ann_ids):
                neighbor_faiss_idx = I[i, 1]  # The second result is the nearest neighbor
                distance = D[i, 1]

                if distance < threshold:
                    neighbor_ann_id = id_map.get(neighbor_faiss_idx)
                    if neighbor_ann_id and neighbor_ann_id in parent:
                        unite_sets(ann_id, neighbor_ann_id)
            
            # Group annotations by their set representative
            groups = {}
            for ann_id in query_ann_ids:
                root = find_set(ann_id)
                if root not in groups:
                    groups[root] = []
                groups[root].append(ann_id)

            copies_to_select = set()
            for root_id, group_ids in groups.items():
                if len(group_ids) > 1:
                    # Sort IDs to consistently pick the same "original".
                    # Sorting strings is reliable.
                    sorted_ids = sorted(group_ids)
                    # The first ID is the original, add the rest to the selection.
                    copies_to_select.update(sorted_ids[1:])
            
            print(f"Found {len(copies_to_select)} duplicate annotations.")
            self.embedding_viewer.render_selection_from_ids(copies_to_select)

        finally:
            QApplication.restoreOverrideCursor()
            
    def find_uncertain_annotations(self):
        """
        Identifies annotations where the model's prediction is uncertain.
        It reuses cached predictions if available, otherwise runs a temporary prediction.
        """
        if not self.embedding_viewer.points_by_id:
            QMessageBox.information(self, "No Data", "Please generate an embedding first.")
            return

        if self.current_embedding_model_info is None:
            QMessageBox.information(self, 
                                    "No Embedding", 
                                    "Could not determine the model used for the embedding. Please run it again.")
            return

        items_in_view = list(self.embedding_viewer.points_by_id.values())
        data_items_in_view = [p.data_item for p in items_in_view]
        
        model_name_from_embedding, feature_mode_from_embedding = self.current_embedding_model_info
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            probabilities_dict = {}

            # Decide whether to reuse cached features or run a new prediction
            if feature_mode_from_embedding == "Predictions":
                print("Reusing cached prediction vectors from the FeatureStore.")
                sanitized_model_name = os.path.basename(model_name_from_embedding).replace(' ', '_').replace('/', '_')
                sanitized_feature_mode = feature_mode_from_embedding.replace(' ', '_').replace('/', '_')
                model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"
                
                probabilities_dict, _ = self.feature_store.get_features(data_items_in_view, model_key)
                if not probabilities_dict:
                    QMessageBox.warning(self, 
                                        "Cache Error", 
                                        "Could not retrieve cached predictions.")
                    return
            else:
                print("Embedding not based on 'Predictions' mode. Running a temporary prediction.")
                model_info_for_predict = self.model_settings_widget.get_selected_model()
                probabilities_dict = self._get_yolo_predictions_for_uncertainty(data_items_in_view, 
                                                                                model_info_for_predict)

            if not probabilities_dict:
                # The helper function will show its own, more specific errors.
                return

            uncertain_ids = []
            params = self.uncertainty_params
            for ann_id, probs in probabilities_dict.items():
                if len(probs) < 2:
                    continue  # Cannot calculate margin

                sorted_probs = np.sort(probs)[::-1]
                top1_conf = sorted_probs[0]
                top2_conf = sorted_probs[1]
                margin = top1_conf - top2_conf

                if top1_conf < params['confidence'] or margin < params['margin']:
                    uncertain_ids.append(ann_id)
            
            self.embedding_viewer.render_selection_from_ids(set(uncertain_ids))
            print(f"Found {len(uncertain_ids)} uncertain annotations.")

        finally:
            QApplication.restoreOverrideCursor()
            
    @pyqtSlot()
    def find_similar_annotations(self):
        """
        Finds k-nearest neighbors to the selected annotation(s) and updates 
        the UI to show the results in an isolated, ordered view. This method
        now ensures the grid is always updated and resets the sort-by dropdown.
        """
        k = self.similarity_params.get('k', 10)

        if not self.annotation_viewer.selected_widgets:
            QMessageBox.information(self, "No Selection", "Please select one or more annotations first.")
            return

        if not self.current_embedding_model_info:
            QMessageBox.warning(self, "No Embedding", "Please run an embedding before searching for similar items.")
            return

        selected_data_items = [widget.data_item for widget in self.annotation_viewer.selected_widgets]
        model_name, feature_mode = self.current_embedding_model_info
        sanitized_model_name = os.path.basename(model_name).replace(' ', '_')
        sanitized_feature_mode = feature_mode.replace(' ', '_').replace('/', '_')
        model_key = f"{sanitized_model_name}_{sanitized_feature_mode}"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            features_dict, _ = self.feature_store.get_features(selected_data_items, model_key)
            if not features_dict:
                QMessageBox.warning(self, 
                                    "Features Not Found", 
                                    "Could not retrieve feature vectors for the selected items.")
                return

            source_vectors = np.array(list(features_dict.values()))
            query_vector = np.mean(source_vectors, axis=0, keepdims=True).astype('float32')

            index = self.feature_store._get_or_load_index(model_key)
            faiss_idx_to_ann_id = self.feature_store.get_faiss_index_to_annotation_id_map(model_key)
            if index is None or not faiss_idx_to_ann_id:
                QMessageBox.warning(self, 
                                    "Index Error", 
                                    "Could not find a valid feature index for the current model.")
                return

            # Find k results, plus more to account for the query items possibly being in the results
            num_to_find = k + len(selected_data_items)
            if num_to_find > index.ntotal:
                num_to_find = index.ntotal
            
            _, I = index.search(query_vector, num_to_find)

            source_ids = {item.annotation.id for item in selected_data_items}
            similar_ann_ids = []
            for faiss_idx in I[0]:
                ann_id = faiss_idx_to_ann_id.get(faiss_idx)
                if ann_id and ann_id in self.data_item_cache and ann_id not in source_ids:
                    similar_ann_ids.append(ann_id)
                if len(similar_ann_ids) == k:
                    break

            # Create the final ordered list: original selection first, then similar items.
            ordered_ids_to_display = list(source_ids) + similar_ann_ids
            
            # --- FIX IMPLEMENTATION ---
            # 1. Force sort combo to "None" to avoid user confusion.
            self.annotation_viewer.sort_combo.setCurrentText("None")

            # 2. Update the embedding viewer selection.
            self.embedding_viewer.render_selection_from_ids(set(ordered_ids_to_display))
            
            # 3. Call the new robust method in AnnotationViewer to handle isolation and grid updates.
            self.annotation_viewer.display_and_isolate_ordered_results(ordered_ids_to_display)

            self.update_button_states()

        finally:
            QApplication.restoreOverrideCursor()
            
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
        Runs dimensionality reduction with automatic PCA preprocessing for UMAP and t-SNE.

        Args:
            features (np.ndarray): Feature matrix of shape (N, D).
            params (dict): Embedding parameters, including technique and its hyperparameters.

        Returns:
            np.ndarray or None: 2D embedded features of shape (N, 2), or None on failure.
        """
        technique = params.get('technique', 'UMAP')
        # Default number of components to use for PCA preprocessing
        pca_components = params.get('pca_components', 50)

        if len(features) <= 2:
            # Not enough samples for dimensionality reduction
            return None

        try:
            # Standardize features before reduction
            features_scaled = StandardScaler().fit_transform(features)
            
            # Apply PCA preprocessing automatically for UMAP or TSNE
            # (only if the feature dimension is larger than the target PCA components)
            if technique in ["UMAP", "TSNE"] and features_scaled.shape[1] > pca_components:
                # Ensure pca_components doesn't exceed number of samples or features
                pca_components = min(pca_components, features_scaled.shape[0] - 1, features_scaled.shape[1])
                print(f"Applying PCA preprocessing to {pca_components} components before {technique}")
                pca = PCA(n_components=pca_components, random_state=42)
                features_scaled = pca.fit_transform(features_scaled)
                variance_explained = sum(pca.explained_variance_ratio_) * 100
                print(f"Variance explained by PCA: {variance_explained:.1f}%")

            # Proceed with the selected dimensionality reduction technique
            if technique == "UMAP":
                n_neighbors = min(params.get('n_neighbors', 15), len(features_scaled) - 1)
                reducer = UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=n_neighbors,
                    min_dist=params.get('min_dist', 0.1),
                    metric=params.get('metric', 'cosine')
                )
            elif technique == "TSNE":
                perplexity = min(params.get('perplexity', 30), len(features_scaled) - 1)
                reducer = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    early_exaggeration=params.get('early_exaggeration', 12.0),
                    learning_rate=params.get('learning_rate', 'auto'),
                    init='pca'
                )
            elif technique == "PCA":
                reducer = PCA(n_components=2, random_state=42)
            else:
                return None

            # Fit and transform the features
            return reducer.fit_transform(features_scaled)

        except Exception as e:
            QMessageBox.warning(self, 
                                "Embedding Error",
                                f"An error occurred during dimensionality reduction with {technique}.\n\nError: {e}")
            
            return None

    def _update_data_items_with_embedding(self, data_items, embedded_features):
        """Updates AnnotationDataItem objects with embedding results."""
        scale_factor = 4000
        min_vals, max_vals = np.min(embedded_features, axis=0), np.max(embedded_features, axis=0)
        range_vals = max_vals - min_vals
        for i, item in enumerate(data_items):
            norm_x = (embedded_features[i, 0] - min_vals[0]) / range_vals[0] if range_vals[0] > 0 else 0.5
            norm_y = (embedded_features[i, 1] - min_vals[1]) / range_vals[1] if range_vals[1] > 0 else 0.5
            item.embedding_x = (norm_x * scale_factor) - (scale_factor / 2)
            item.embedding_y = (norm_y * scale_factor) - (scale_factor / 2)
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

            progress_bar.set_busy_mode("Running dimensionality reduction...")
            embedded_features = self._run_dimensionality_reduction(features, embedding_params)

            if embedded_features is None:
                return

            progress_bar.set_busy_mode("Updating visualization...")
            self._update_data_items_with_embedding(self.current_data_items, embedded_features)
            self.embedding_viewer.update_embeddings(self.current_data_items)
            self.embedding_viewer.show_embedding()
            self.embedding_viewer.fit_view_to_points()

            # Check if confidence scores are available to enable sorting
            _, feature_mode = self.current_embedding_model_info
            is_predict_mode = feature_mode == "Predictions"
            self.annotation_viewer.set_confidence_sort_availability(is_predict_mode)

            # If using Predictions mode, update data items with probabilities for confidence sorting
            if is_predict_mode:
                for item in self.current_data_items:
                    if item.annotation.id in cached_features:
                        item.prediction_probabilities = cached_features[item.annotation.id]

            # When a new embedding is run, any previous similarity sort becomes irrelevant
            self.annotation_viewer.active_ordered_ids = []

        finally:
            QApplication.restoreOverrideCursor()
            progress_bar.finish_progress()
            progress_bar.stop_progress()
            progress_bar.close()

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
            self.annotation_viewer.set_confidence_sort_availability(False)
            
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

            # 6. Refresh main window annotations list
            affected_images = {ann.image_path for ann in annotations_to_delete_from_main_app}
            for image_path in affected_images:
                self.image_window.update_image_annotations(image_path)
            self.annotation_window.load_annotations()

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

        # Logic for the "Find Similar" button
        selection_exists = bool(self.annotation_viewer.selected_widgets)
        embedding_exists = bool(self.embedding_viewer.points_by_id) and self.current_embedding_model_info is not None
        self.annotation_viewer.find_similar_button.setEnabled(selection_exists and embedding_exists)

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
            affected_images = {ann.image_path for ann in applied_label_changes}
            for image_path in affected_images:
                self.image_window.update_image_annotations(image_path)
            self.annotation_window.load_annotations()

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
            
    def _cleanup_resources(self):
        """Clean up resources."""
        self.loaded_model = None
        self.model_path = ""
        self.current_features = None
        self.current_feature_generating_model = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()