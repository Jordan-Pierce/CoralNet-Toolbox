"""
Selection Model for MVAT

Centralized state manager for camera selections. Enforces the UI paradigm
where there is one 'Active' camera, and a set of 'Selected' cameras.
The Active camera is strictly enforced to always be within the Selected set.
"""

from PyQt5.QtCore import QObject, pyqtSignal

class SelectionModel(QObject):
    """
    Single source of truth for camera selection state.
    """
    
    # Emitted when the primary active camera changes (e.g., for 3D viewer jumps)
    # Sends the image_path (str)
    active_changed = pyqtSignal(str)       
    
    # Emitted when the set of selected/highlighted cameras changes
    # Sends a set of image_paths (set)
    selection_changed = pyqtSignal(set)    

    def __init__(self, parent=None):
        """Initialize the selection model."""
        super().__init__(parent)
        self._active_path = None
        self._selected_paths = set()

    @property
    def active_path(self):
        """Get the current active camera path."""
        return self._active_path

    @property
    def selected_paths(self):
        """Get the set of all selected camera paths."""
        return self._selected_paths

    def set_active(self, path):
        """
        Set the primary active camera (e.g., via Double-Click).
        Automatically ensures it is added to the selected set.
        """
        if self._active_path == path:
            return # No change needed

        self._active_path = path
        self._selected_paths.add(path)
        
        self.active_changed.emit(self._active_path)
        self.selection_changed.emit(self._selected_paths)

    def toggle_selection(self, path):
        """
        Toggle the secondary selection state for a camera (e.g., via Ctrl+Click).
        Prevents toggling off the active camera.
        """
        # Rule: You cannot toggle off the active camera
        if path == self._active_path:
            return 

        if path in self._selected_paths:
            self._selected_paths.remove(path)
        else:
            self._selected_paths.add(path)
            
        self.selection_changed.emit(self._selected_paths)

    def add_selection(self, path):
        """Explicitly add a camera to the selection set."""
        if path not in self._selected_paths:
            self._selected_paths.add(path)
            self.selection_changed.emit(self._selected_paths)

    def remove_selection(self, path):
        """Explicitly remove a camera from the selection set."""
        if path == self._active_path:
            return  # Protect the active camera
            
        if path in self._selected_paths:
            self._selected_paths.remove(path)
            self.selection_changed.emit(self._selected_paths)

    def set_selections(self, paths_iterable):
        """
        Overwrite the current selections (e.g., Shift+Click range, or Ctrl+A).
        Guarantees the active camera remains in the set.
        """
        self._selected_paths = set(paths_iterable)
        
        if self._active_path:
            self._selected_paths.add(self._active_path)
            
        self.selection_changed.emit(self._selected_paths)

    def clear_selections(self):
        """
        Clear all secondary selections (e.g., via Escape key).
        Leaves only the active camera selected.
        """
        if self._active_path:
            self._selected_paths = {self._active_path}
        else:
            self._selected_paths = set()
            
        self.selection_changed.emit(self._selected_paths)
        
    def clear_all(self):
        """
        Completely clear both active and selected states.
        Useful when tearing down the view or unloading a project.
        """
        self._active_path = None
        self._selected_paths = set()
        
        # Emit empty string or None depending on how slots handle it.
        # Using None implies no active camera.
        self.active_changed.emit("") 
        self.selection_changed.emit(self._selected_paths)