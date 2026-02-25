# coralnet_toolbox/Explorer/workers/QtEmbeddingWorker.py
"""
Background worker for running the embedding pipeline without blocking the UI.

This module provides a QThread-based worker that handles the expensive
feature extraction and dimensionality reduction operations in the background,
keeping the UI responsive.
"""
import warnings
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPipelineWorker(QThread):
    """
    Worker thread for running the embedding pipeline in the background.
    
    This worker handles:
    - Feature extraction from annotation crops
    - Feature caching
    - Dimensionality reduction
    
    Signals:
        progress (str): Emitted with status updates
        finished (object): Emitted when complete with results dict
        error (str): Emitted if an error occurs
    """
    
    progress = pyqtSignal(str)  # Status message
    finished = pyqtSignal(object)  # Results dict
    error = pyqtSignal(str)  # Error message
    
    def __init__(self, 
                 data_items,
                 model_name,
                 model_key,
                 embedding_params,
                 cache_manager,
                 feature_extractor_fn,
                 dim_reduction_fn):
        """
        Initialize the worker.
        
        Args:
            data_items: List of AnnotationDataItem objects
            model_name: Name/path of the feature extraction model
            model_key: Cache key for the model
            embedding_params: Dict of dimensionality reduction parameters
            cache_manager: CacheManager instance for feature caching
            feature_extractor_fn: Callable that extracts features (model_name, data_items) -> features, valid_items
            dim_reduction_fn: Callable that runs dimensionality reduction (features, params) -> embedded_features
        """
        super().__init__()
        self.data_items = data_items
        self.model_name = model_name
        self.model_key = model_key
        self.embedding_params = embedding_params
        self.cache_manager = cache_manager
        self.feature_extractor_fn = feature_extractor_fn
        self.dim_reduction_fn = dim_reduction_fn
        self._cancelled = False
        
    def cancel(self):
        """Request cancellation of the pipeline."""
        self._cancelled = True
        
    def run(self):
        """Execute the pipeline in the background."""
        try:
            if self._cancelled:
                return
                
            # Check feature cache
            self.progress.emit("Checking feature cache...")
            cached_features, items_to_process = self.cache_manager.get_features(
                self.data_items, self.model_key
            )
            
            if self._cancelled:
                return
            
            # Extract features for uncached items
            if items_to_process:
                self.progress.emit(f"Extracting features for {len(items_to_process)} annotations...")
                newly_extracted_features, valid_items = self.feature_extractor_fn(
                    self.model_name, items_to_process
                )
                
                if self._cancelled:
                    return
                
                if len(newly_extracted_features) > 0:
                    self.progress.emit("Saving features to cache...")
                    self.cache_manager.add_features(
                        valid_items, newly_extracted_features, self.model_key
                    )
                    for item, vec in zip(valid_items, newly_extracted_features):
                        cached_features[item.annotation.id] = vec
            
            if self._cancelled:
                return
            
            if not cached_features:
                self.error.emit("No features could be extracted.")
                return
            
            # Assemble final feature matrix
            self.progress.emit("Assembling feature matrix...")
            final_features = []
            final_data_items = []
            missing_features = []
            
            for item in self.data_items:
                if item.annotation.id in cached_features:
                    final_features.append(cached_features[item.annotation.id])
                    final_data_items.append(item)
                else:
                    missing_features.append(item.annotation.id)
            
            if self._cancelled:
                return
            
            # Check for missing features
            if missing_features:
                self.error.emit(
                    f"{len(missing_features)} annotation(s) could not have features extracted.\n"
                    "All annotations must have features to display the embedding."
                )
                return
            
            features = np.array(final_features)
            
            if self._cancelled:
                return
            
            # Run dimensionality reduction
            self.progress.emit("Running dimensionality reduction...")
            embedded_features = self.dim_reduction_fn(features, self.embedding_params)
            
            if self._cancelled:
                return
            
            if embedded_features is None:
                self.error.emit("Dimensionality reduction failed.")
                return
            
            # Success - emit results
            self.progress.emit("Complete!")
            results = {
                'data_items': final_data_items,
                'features': features,
                'embedded_features': embedded_features,
                'model_key': self.model_key
            }
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"Pipeline error: {str(e)}")
