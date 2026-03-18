# coralnet_toolbox/Explorer/QtSelectionManager.py
"""
Centralized selection management for Explorer windows.

This module provides the SelectionManager class that handles selection
synchronization between AnnotationViewerWindow, EmbeddingViewerWindow,
and AnnotationWindow. It restores the selection logic from the original
ExplorerWindow implementation.
"""

import warnings
import time

from typing import List, Set

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectionManager(QObject):
    """
    Centralized selection manager that synchronizes selection state across
    the AnnotationViewerWindow, EmbeddingViewerWindow, and AnnotationWindow.
    
    This class implements the selection logic from the original ExplorerWindow,
    ensuring consistent behavior when users select annotations in any viewer.
    
    Key behaviors:
    - Single selection: Update label window to that annotation's label
    - Multiple selection: Clear active label in label window
    - Click on annotation/point: Switch to Select tool
    - Don't update Confidence window for multiple selections
    - Sync selection across all viewers (gallery, embedding, annotation window)
    
    Signals:
        selection_changed (list): Emitted when the master selection changes.
        label_window_update_requested (str or None): Emitted to update label window.
    """
    
    selection_changed = pyqtSignal(list)  # List of selected annotation IDs
    label_window_update_requested = pyqtSignal(object)  # Label to select or None to deselect
    
    def __init__(self, main_window, parent=None):
        """
        Initialize the SelectionManager.
        
        Args:
            main_window: Reference to the MainWindow instance.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.main_window = main_window
        
        # References to managed windows (set after they're created)
        self._annotation_viewer = None
        self._embedding_viewer = None
        self._annotation_window = None
        self._label_window = None
        self._confidence_window = None
        
        # Sync flags to prevent infinite loops
        self._syncing = False
        self._block_label_updates = False
        
        # Master selection state (annotation IDs)
        self._selected_ids: Set[str] = set()
        
    # -------------------------------------------------------------------------
    # Window Registration
    # -------------------------------------------------------------------------
    
    def register_annotation_viewer(self, viewer):
        """
        Register the AnnotationViewerWindow and connect signals.
        
        Args:
            viewer: AnnotationViewerWindow instance.
        """
        self._annotation_viewer = viewer
        if viewer:
            viewer.selection_changed.connect(self._on_annotation_viewer_selection_changed)
            viewer.reset_view_requested.connect(self._on_reset_view_requested)
    
    def register_embedding_viewer(self, viewer):
        """
        Register the EmbeddingViewerWindow and connect signals.
        
        Args:
            viewer: EmbeddingViewerWindow instance.
        """
        self._embedding_viewer = viewer
        if viewer:
            viewer.selection_changed.connect(self._on_embedding_viewer_selection_changed)
            viewer.reset_view_requested.connect(self._on_reset_view_requested)
    
    def register_annotation_window(self, window):
        """
        Register the AnnotationWindow and connect signals.
        
        Args:
            window: AnnotationWindow instance.
        """
        self._annotation_window = window
        if window:
            window.annotationSelectionChanged.connect(self._on_annotation_window_selection_changed)
    
    def register_label_window(self, window):
        """
        Register the LabelWindow.
        
        Args:
            window: LabelWindow instance.
        """
        self._label_window = window
    
    def register_confidence_window(self, window):
        """
        Register the ConfidenceWindow.
        
        Args:
            window: ConfidenceWindow instance.
        """
        self._confidence_window = window
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def get_selected_ids(self) -> List[str]:
        """Get the list of currently selected annotation IDs."""
        return list(self._selected_ids)
    
    def clear_selection(self):
        """Clear all selections across all viewers."""
        if self._syncing:
            return
        
        self._syncing = True
        try:
            self._selected_ids.clear()
            
            # Clear annotation viewer
            if self._annotation_viewer:
                self._annotation_viewer.clear_selection()
            
            # Clear embedding viewer
            if self._embedding_viewer:
                self._embedding_viewer.render_selection_from_ids(set())
            
            # Clear annotation window
            if self._annotation_window:
                self._annotation_window.unselect_annotations()
            
            # Update label window
            self._update_label_window_selection()
            
            self.selection_changed.emit([])
        finally:
            self._syncing = False
    
    def select_annotations(self, annotation_ids: List[str], source: str = 'external'):
        """
        Select annotations by ID, syncing across all viewers.
        
        Args:
            annotation_ids: List of annotation IDs to select.
            source: Source of the selection ('gallery', 'embedding', 'annotation', 'external')
        """
        if self._syncing:
            return
                
        self._syncing = True
        try:
            self._selected_ids = set(annotation_ids)
            
            # Sync to annotation viewer (if not the source)
            if source != 'gallery' and self._annotation_viewer:
                if self._selected_ids:
                    # Isolate the selection so they are grouped together automatically
                    self._annotation_viewer.isolate_and_select_from_ids(self._selected_ids)
                else:
                    self._annotation_viewer.clear_selection()
            
            # Sync to embedding viewer (if not the source)
            if source != 'embedding' and self._embedding_viewer:
                self._embedding_viewer.render_selection_from_ids(self._selected_ids)
            
            # Sync to annotation window (if not the source)
            if source != 'annotation' and self._annotation_window:
                self._sync_annotation_window_selection(annotation_ids)
            
            # Switch to Select tool when selecting annotations
            self._switch_to_select_tool()
            
            # Update label window based on selection
            self._update_label_window_selection()
            
            # Update confidence window (only for single selection)
            self._update_confidence_window()
            
            self.selection_changed.emit(list(self._selected_ids))
        finally:
            self._syncing = False
    
    # -------------------------------------------------------------------------
    # Signal Handlers
    # -------------------------------------------------------------------------
    
    @pyqtSlot(list)
    def _on_annotation_viewer_selection_changed(self, changed_ann_ids: List[str]):
        """
        Handle selection changes from AnnotationViewerWindow.
        
        Syncs selection to embedding viewer and annotation window.
        
        Args:
            changed_ann_ids: List of annotation IDs that changed.
        """
        if self._syncing:
            return
        
        self._syncing = True
        try:
            # Get all selected IDs from the annotation viewer
            if self._annotation_viewer:
                all_selected_ids = self._annotation_viewer.get_selected_annotation_ids()
            else:
                all_selected_ids = []
            
            self._selected_ids = set(all_selected_ids)
            
            # Sync selection to the embedding viewer
            if self._embedding_viewer and self._embedding_viewer.points_by_id:
                self._embedding_viewer.render_selection_from_ids(self._selected_ids)
            
            # Sync selection to the annotation window
            if self._annotation_window:
                self._sync_annotation_window_selection(all_selected_ids)
            
            # Switch to Select tool
            self._switch_to_select_tool()
            
            # Update the label window based on the new selection
            self._update_label_window_selection()
            
            # Update confidence window
            self._update_confidence_window()
            
            self.selection_changed.emit(list(self._selected_ids))
            
        finally:
            self._syncing = False
    
    @pyqtSlot(list)
    def _on_embedding_viewer_selection_changed(self, all_selected_ann_ids: List[str]):
        """
        Handle selection changes from EmbeddingViewerWindow.
        
        Syncs selection to annotation viewer (with isolation) and annotation window.
        
        Note: When selection is cleared via single-click, viewers stay in their current
        state (isolated or not). Double-click triggers reset_view_requested to exit isolation.
        
        Args:
            all_selected_ann_ids: List of all selected annotation IDs.
        """
        if self._syncing:
            return
        
        self._syncing = True
        try:
            selected_ids_set = set(all_selected_ann_ids)
            self._selected_ids = selected_ids_set
            
            # If a selection is made in the embedding viewer, isolate those widgets
            if selected_ids_set and self._annotation_viewer:
                # This method handles setting the isolated set and selecting them
                self._annotation_viewer.isolate_and_select_from_ids(selected_ids_set)
            elif not selected_ids_set and self._annotation_viewer:
                # Selection was cleared - just clear selection but DON'T exit isolation mode
                # (User can double-click to exit isolation mode via reset_view_requested)
                self._annotation_viewer.clear_selection()

            # Sync to annotation window synchronously to maintain the syncing lock.
            if self._annotation_window:
                self._sync_annotation_window_selection(all_selected_ann_ids)
            
            # Switch to Select tool
            self._switch_to_select_tool()
            
            # Update the label window based on the selection
            self._update_label_window_selection()
            
            # Update confidence window
            self._update_confidence_window()
            
            self.selection_changed.emit(list(self._selected_ids))
            
        finally:
            self._syncing = False
    
    @pyqtSlot(object)
    def _on_annotation_window_selection_changed(self, selected_ids):
        """
        Handle selection changes from AnnotationWindow.
        
        Syncs selection to both viewer windows.
        
        Args:
            selected_ids: List of annotation IDs that are now selected.
        """
        if self._syncing:
            return
        
        self._syncing = True
        try:
            if selected_ids:
                self._selected_ids = set(selected_ids)
            else:
                self._selected_ids.clear()
            
            # Sync to annotation viewer
            if self._annotation_viewer:
                if self._selected_ids:
                    # Automatically isolate the canvas selections in the gallery
                    self._annotation_viewer.isolate_and_select_from_ids(self._selected_ids)
                else:
                    # If selection is cleared on canvas, clear it here too 
                    # (Note: double-click the gallery to exit isolation mode)
                    self._annotation_viewer.clear_selection()
            
            # Sync to embedding viewer
            if self._embedding_viewer:
                self._embedding_viewer.render_selection_from_ids(self._selected_ids)
            
            # Update label window
            self._update_label_window_selection()
            
            # Update confidence window
            self._update_confidence_window()
            
            self.selection_changed.emit(list(self._selected_ids))
            
        finally:
            self._syncing = False
    
    @pyqtSlot()
    def _on_reset_view_requested(self):
        """
        Handle reset view requests from double-click in either viewer.
        
        Clears selections and exits isolation mode in both viewers.
        """
        if self._syncing:
            return
        
        self._syncing = True
        try:
            self._selected_ids.clear()
            
            # Clear all selections in both viewers
            if self._annotation_viewer:
                self._annotation_viewer.clear_selection()
                # Exit isolation mode if currently active
                if self._annotation_viewer.isolated_mode:
                    self._annotation_viewer._show_all_annotations()
                # Clear similarity sort context
                self._annotation_viewer.active_ordered_ids = []
            
            if self._embedding_viewer:
                self._embedding_viewer.render_selection_from_ids(set())
                # Exit isolation mode if currently active
                if self._embedding_viewer.isolated_mode:
                    self._embedding_viewer._show_all_points()
            
            # Clear annotation window selection
            if self._annotation_window:
                self._annotation_window.unselect_annotations()
            
            # Update label window
            self._update_label_window_selection()
            
            self.selection_changed.emit([])
            
        finally:
            self._syncing = False
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _sync_annotation_window_selection(self, annotation_ids: List[str]):
        """
        Sync selection to AnnotationWindow.
        
        Args:
            annotation_ids: List of annotation IDs to select.
        """
        if not self._annotation_window:
            return
        
        # Use the batched selection API on AnnotationWindow when available
        try:
            if hasattr(self._annotation_window, 'select_annotations_by_ids'):
                self._annotation_window.select_annotations_by_ids(annotation_ids, scroll_to_first=True, quiet_mode=True)
            else:
                # Fallback to older per-item selection
                self._annotation_window.unselect_annotations()
                if annotation_ids:
                    annotations_dict = getattr(self._annotation_window, 'annotations_dict', {})
                    for ann_id in annotation_ids:
                        if ann_id in annotations_dict:
                            ann = annotations_dict[ann_id]
                            self._annotation_window.select_annotation(ann, multi_select=True, quiet_mode=True)
        except Exception:
            # Ensure we don't crash selection manager on unexpected errors
            try:
                self._annotation_window.unselect_annotations()
            except Exception:
                pass
    
    def _switch_to_select_tool(self):
        """Switch to the Select tool when annotations are selected, preserving the selection.
        
        This is called after selection synchronization completes. The preserve_selection
        parameter ensures that the synchronized selections are not cleared by the tool switch.
        """
        if not self._selected_ids:
            return
        
        # Check if already on select tool - if so, no need to switch
        if hasattr(self.main_window, 'annotation_window'):
            current_tool = self.main_window.annotation_window.get_selected_tool()
            if current_tool == 'select':
                return  # Already on select tool
        
        # Switch to select tool with preserve_selection=True to keep synchronized selections
        if hasattr(self.main_window, 'choose_specific_tool'):
            self.main_window.choose_specific_tool('select', preserve_selection=True)
    
    def _update_label_window_selection(self):
        """
        Update the label window based on the current selection state.
        
        This is the single, centralized point of logic for label window updates:
        - Single selection: Set active label to that annotation's label
        - Multiple selection: Deselect active label
        - No selection: Keep active label (don't interfere with normal annotation workflow)
        """
        if not self._label_window or self._block_label_updates:
            return
        
        if not self._selected_ids:
            # No selection - don't deselect active label
            # This preserves the user's selected label when switching tools or clearing selections
            self._label_window.update_annotation_count()
            return
        
        # Get data items for selected annotations
        selected_data_items = self._get_selected_data_items()
        
        if not selected_data_items:
            self._label_window.deselect_active_label()
            self._label_window.update_annotation_count()
            return
        
        # Get the effective label from the first selected item
        first_label = selected_data_items[0].effective_label
        
        # Check if all selected items have the same label
        all_same_label = all(
            item.effective_label.id == first_label.id
            for item in selected_data_items
        )
        
        if len(selected_data_items) == 1 or all_same_label:
            # Single selection or all same label - set active label
            label_widget = self._label_window.get_label_by_id(first_label.id, return_review=True)
            if label_widget:
                self._label_window.set_active_label(label_widget)
                # Emit signal to update other UI elements
                if self._annotation_window:
                    self._annotation_window.labelSelected.emit(first_label.id)
            else:
                self._label_window.deselect_active_label()
        else:
            # Multiple different labels selected - deselect active label
            self._label_window.deselect_active_label()
        
        self._label_window.update_annotation_count()
    
    def _update_confidence_window(self):
        """
        Update the Confidence window based on current selection.
        
        Only updates for single selection to avoid showing misleading info.
        """
        if not self._confidence_window:
            return
        
        # Only update for single selection
        if len(self._selected_ids) != 1:
            return
        
        # Get the single selected annotation
        ann_id = list(self._selected_ids)[0]
        
        # Get the annotation from annotation window
        if self._annotation_window and hasattr(self._annotation_window, 'annotations_dict'):
            ann = self._annotation_window.annotations_dict.get(ann_id)
            if ann and hasattr(self._confidence_window, 'update_for_annotation'):
                self._confidence_window.update_for_annotation(ann)
    
    def _get_selected_data_items(self):
        """
        Get data items for all selected annotations.
        
        Tries to get from annotation viewer cache first, then embedding viewer.
        
        Returns:
            List of AnnotationDataItem or similar objects with effective_label property.
        """
        data_items = []
        
        # Try annotation viewer cache first
        if self._annotation_viewer:
            cache = getattr(self._annotation_viewer, 'data_item_cache', {})
            for ann_id in self._selected_ids:
                if ann_id in cache:
                    data_items.append(cache[ann_id])
        
        # If not found, try embedding viewer cache
        if not data_items and self._embedding_viewer:
            cache = getattr(self._embedding_viewer, 'data_item_cache', {})
            for ann_id in self._selected_ids:
                if ann_id in cache:
                    data_items.append(cache[ann_id])
        
        # Last resort: create wrapper objects from annotations
        if not data_items and self._annotation_window:
            annotations_dict = getattr(self._annotation_window, 'annotations_dict', {})
            for ann_id in self._selected_ids:
                if ann_id in annotations_dict:
                    ann = annotations_dict[ann_id]
                    # Create a simple wrapper with effective_label
                    class SimpleDataItem:
                        def __init__(self, annotation):
                            self.annotation = annotation
                            self.effective_label = annotation.label
                    data_items.append(SimpleDataItem(ann))
        
        return data_items
    
    # -------------------------------------------------------------------------
    # Context Menu Support
    # -------------------------------------------------------------------------
    
    def handle_context_menu_selection(self, annotation_id: str, navigate_to: bool = True):
        """
        Handle Ctrl+Right-click context menu selection.
        
        Selects the annotation and optionally navigates to it in AnnotationWindow.
        
        Args:
            annotation_id: The annotation ID to select.
            navigate_to: Whether to navigate to and center on the annotation.
        """
        if self._syncing:
            return
        
        self._syncing = True
        try:
            self._selected_ids = {annotation_id}
            
            # Sync to annotation viewer
            if self._annotation_viewer:
                if self._selected_ids:
                    self._annotation_viewer.isolate_and_select_from_ids(self._selected_ids)
                else:
                    self._annotation_viewer.clear_selection()
            
            # Sync to embedding viewer
            if self._embedding_viewer:
                self._embedding_viewer.render_selection_from_ids(self._selected_ids)
            
            # Navigate to annotation in annotation window
            if navigate_to and self._annotation_window:
                annotations_dict = getattr(self._annotation_window, 'annotations_dict', {})
                if annotation_id in annotations_dict:
                    ann = annotations_dict[annotation_id]
                    
                    # Change image if needed
                    if self._annotation_window.current_image_path != ann.image_path:
                        if hasattr(self._annotation_window, 'set_image'):
                            self._annotation_window.set_image(ann.image_path)
                    
                    # Select and center on annotation
                    if hasattr(self._annotation_window, 'select_annotation'):
                        self._annotation_window.select_annotation(ann, quiet_mode=True)
                    if hasattr(self._annotation_window, 'center_on_annotation'):
                        self._annotation_window.center_on_annotation(ann)
            
            # Update label window
            self._update_label_window_selection()
            
            # Update confidence window
            self._update_confidence_window()
            
            self.selection_changed.emit([annotation_id])
            
        finally:
            self._syncing = False
