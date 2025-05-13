import warnings

from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.Rasters import RasterManager

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ImageFilter(QObject):
    """
    Handles filtering of images based on various criteria.
    """
    # Signals
    filteringStarted = pyqtSignal()
    filteringFinished = pyqtSignal(list)  # Filtered paths
    filterProgress = pyqtSignal(int, int)  # Current, total
    
    def __init__(self, raster_manager: RasterManager):
        """
        Initialize the ImageFilter.
        
        Args:
            raster_manager (RasterManager): Reference to the raster manager
        """
        super().__init__()
        self.raster_manager = raster_manager
        self._is_filtering = False
        
    def filter_images(self, 
                      search_text: str = "",
                      search_label: str = "",
                      require_annotations: bool = False,
                      require_no_annotations: bool = False,
                      require_predictions: bool = False,
                      selected_paths: List[str] = None,
                      use_threading: bool = True,
                      callback: Callable = None) -> List[str]:
        """
        Filter images based on various criteria.
        
        Args:
            search_text (str): Text to search for in image names
            search_label (str): Label code to search for
            require_annotations (bool): Require images to have annotations
            require_no_annotations (bool): Require images to have no annotations
            require_predictions (bool): Require images to have predictions
            selected_paths (List[str]): List of paths to filter from
            use_threading (bool): Whether to use multithreading
            callback (Callable): Optional callback function to call when filtering is complete
            
        Returns:
            List[str]: List of filtered image paths
        """
        # If no filters are active, return all paths
        if not any([search_text, 
                    search_label, 
                    require_annotations, 
                   require_no_annotations, 
                   require_predictions, 
                   selected_paths]):
            result = self.raster_manager.image_paths.copy()
            self.filteringStarted.emit()
            self.filteringFinished.emit(result)
            if callback:
                callback(result)
            return result
            
        # Use threading if requested
        if use_threading:
            return self._filter_with_threading(
                search_text, search_label, require_annotations,
                require_no_annotations, require_predictions, selected_paths, callback
            )
        else:
            return self._filter_images_sync(
                search_text, search_label, require_annotations, 
                require_no_annotations, require_predictions, selected_paths, callback
            )
    
    def _filter_images_sync(self, 
                            search_text: str,
                            search_label: str,
                            require_annotations: bool,
                            require_no_annotations: bool,
                            require_predictions: bool,
                            selected_paths: List[str],
                            callback: Callable = None) -> List[str]:
        """
        Filter images synchronously.
        
        Args: Same as filter_images but without use_threading
            
        Returns:
            List[str]: Filtered image paths
        """
        self._is_filtering = True
        self.filteringStarted.emit()
        
        # Get filtered paths from raster manager
        filtered_paths = self.raster_manager.get_filtered_paths(
            search_text=search_text,
            search_label=search_label,
            require_annotations=require_annotations,
            require_no_annotations=require_no_annotations,
            require_predictions=require_predictions,
            selected_paths=selected_paths
        )
        
        # Sort the filtered paths
        filtered_paths.sort()
        
        self._is_filtering = False
        self.filteringFinished.emit(filtered_paths)
        
        if callback:
            callback(filtered_paths)
            
        return filtered_paths
    
    def _filter_with_threading(self, 
                               search_text: str,
                               search_label: str,
                               require_annotations: bool,
                               require_no_annotations: bool,
                               require_predictions: bool,
                               selected_paths: List[str],
                               callback: Callable = None) -> List[str]:
        """
        Filter images using multiple threads.
        
        Args: Same as filter_images but without use_threading
            
        Returns:
            List[str]: Filtered image paths
        """
        self._is_filtering = True
        self.filteringStarted.emit()
        
        filtered_paths = []
        
        # Get paths to filter
        paths_to_filter = self.raster_manager.image_paths
        total_paths = len(paths_to_filter)
        
        # Use a thread pool to filter in parallel
        with ThreadPoolExecutor() as executor:
            # Submit all filtering tasks
            futures = {}
            for i, path in enumerate(paths_to_filter):
                # Skip if not in selected paths
                if selected_paths is not None and path not in selected_paths:
                    continue
                    
                # Get raster from manager
                raster = self.raster_manager.get_raster(path)
                if not raster:
                    continue
                    
                # Submit filtering task
                future = executor.submit(
                    raster.matches_filter,
                    search_text=search_text,
                    search_label=search_label,
                    require_annotations=require_annotations,
                    require_no_annotations=require_no_annotations,
                    require_predictions=require_predictions
                )
                futures[future] = path
            
            # Process results as they complete
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                completed += 1
                
                # Emit progress
                self.filterProgress.emit(completed, total_paths)
                
                # Add path if it matches filter
                if future.result():
                    filtered_paths.append(path)
        
        # Sort the filtered paths
        filtered_paths.sort()
        
        self._is_filtering = False
        self.filteringFinished.emit(filtered_paths)
        
        if callback:
            callback(filtered_paths)
            
        return filtered_paths
    
    @property
    def is_filtering(self) -> bool:
        """Get whether filtering is in progress."""
        return self._is_filtering