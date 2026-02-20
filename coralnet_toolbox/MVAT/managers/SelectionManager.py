"""
Selection Manager for MVAT

Centralized state manager for camera selections. Enforces the UI paradigm
where there is one 'Active' camera, and a set of 'Selected' cameras.
The Active camera is strictly enforced to always be within the Selected set.
"""

from PyQt5.QtCore import QObject, pyqtSignal


class SelectionManager(QObject):
    """
    Single source of truth for camera selection state.
    """
    
    # Emitted when the primary active camera changes (e.g., for 3D viewer jumps)
    # Sends the image_path (str)
    active_changed = pyqtSignal(str)       
    
    # Emitted when the set of selected/highlighted cameras changes
    # Sends a set of image_paths (set)
    selection_changed = pyqtSignal(set)    
    
    # Optional: last clicked path (useful for shift-range operations)
    last_clicked_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the selection manager."""
        super().__init__(parent)
        self._active_path = None
        self._selected_paths = set()
        # Last clicked path for range-selection support
        self._last_clicked = None

    @property
    def active_path(self):
        """Get the current active camera path."""
        return self._active_path

    @property
    def selected_paths(self):
        """Get the set of all selected camera paths."""
        return self._selected_paths

    # -----------------------
    # Convenience queries
    # -----------------------
    def is_active(self, path: str) -> bool:
        """Return True if the given path is the active camera."""
        return path == self._active_path

    def is_selected(self, path: str) -> bool:
        """Return True if the given path is in the selected set."""
        return path in self._selected_paths

    def get_selected_list(self, ordered_paths: list = None) -> list:
        """Return selected paths as a list.

        If ordered_paths is provided, selection order will follow that list.
        """
        if ordered_paths:
            return [p for p in ordered_paths if p in self._selected_paths]
        return list(self._selected_paths)

    def set_active(self, path, emit: bool = True):
        """Set the primary active camera and ensure it's selected.

        Args:
            path: image path to become active (or None/"" to clear)
            emit: whether to emit signals (use False during bulk updates)
        """
        if self._active_path == path:
            return

        self._active_path = path
        if path is not None and path != "":
            self._selected_paths.add(path)

        if emit:
            # Keep API backward-compat: emit a string (empty when None)
            self.active_changed.emit(self._active_path or "")
            self.selection_changed.emit(self._selected_paths)

    def toggle_selection(self, path):
        """
        Toggle the secondary selection state for a camera (e.g., via Ctrl+Click).
        Prevents toggling off the active camera.
        """
        # Rule: You cannot toggle off the active camera
        if path == self._active_path:
            return False

        if path in self._selected_paths:
            self._selected_paths.remove(path)
            new_state = False
        else:
            self._selected_paths.add(path)
            new_state = True

        self.selection_changed.emit(self._selected_paths)
        return new_state

    # New alias with return value
    def toggle(self, path, emit: bool = True) -> bool:
        """Toggle selection and return resulting selected state."""
        result = self.toggle_selection(path)
        # toggle_selection already emits; honor emit flag by re-emitting if needed
        if emit and result is not None:
            self.selection_changed.emit(self._selected_paths)
        return bool(result)

    def add_selection(self, path):
        """Explicitly add a camera to the selection set."""
        if path not in self._selected_paths:
            self._selected_paths.add(path)
            self.selection_changed.emit(self._selected_paths)

    def select(self, path, emit: bool = True):
        """Select a path (same as add_selection, supports emit control)."""
        if path not in self._selected_paths:
            self._selected_paths.add(path)
        if emit:
            self.selection_changed.emit(self._selected_paths)

    def remove_selection(self, path):
        """Explicitly remove a camera from the selection set."""
        if path == self._active_path:
            return  # Protect the active camera

        if path in self._selected_paths:
            self._selected_paths.remove(path)
            self.selection_changed.emit(self._selected_paths)

    def deselect(self, path, force: bool = False, emit: bool = True):
        """Deselect a path. If force=True, allow deselecting the active camera.

        If the active camera is force-removed, pick a new active.
        """
        if path == self._active_path and not force:
            return

        if path in self._selected_paths:
            self._selected_paths.remove(path)

        if path == self._active_path and force:
            # Active removed: promote another
            self.pick_next_active_on_remove()

        if emit:
            self.selection_changed.emit(self._selected_paths)

    def set_selections(self, paths_iterable, emit: bool = True):
        """
        Overwrite the current selections (e.g., Shift+Click range, or Ctrl+A).
        Guarantees the active camera remains in the set.
        """
        self._selected_paths = set(paths_iterable)

        # Guarantee active remains selected
        if self._active_path:
            self._selected_paths.add(self._active_path)

        if emit:
            self.selection_changed.emit(self._selected_paths)

    def clear_selections(self, keep_active: bool = True, emit: bool = True):
        """Clear selections.

        Args:
            keep_active: if True, leave active camera selected; otherwise clear all.
            emit: whether to emit selection_changed
        """
        if keep_active and self._active_path:
            self._selected_paths = {self._active_path}
        else:
            self._selected_paths = set()

        if emit:
            self.selection_changed.emit(self._selected_paths)
        
    def clear_all(self):
        """
        Completely clear both active and selected states.
        Useful when tearing down the view or unloading a project.
        """
        self._active_path = None
        self._selected_paths = set()
        self._last_clicked = None

        # Emit empty string for backward compatibility
        self.active_changed.emit("")
        self.selection_changed.emit(self._selected_paths)

    # -----------------------
    # Batch and helper operations
    # -----------------------
    def select_all(self, available_paths: list, emit: bool = True):
        """Select all provided available_paths."""
        self._selected_paths = set(available_paths)
        if self._active_path:
            self._selected_paths.add(self._active_path)
        if emit:
            self.selection_changed.emit(self._selected_paths)

    def remove_paths(self, paths_iterable: list, available_paths: list = None, emit: bool = True):
        """Remove a set of paths (e.g., when cameras are unloaded).

        If the active camera is removed, promote a sensible new active.
        """
        removed = set(paths_iterable)
        # Remove from selected set
        self._selected_paths.difference_update(removed)

        active_removed = self._active_path in removed
        if active_removed:
            # Choose next active from remaining selected, otherwise from available_paths
            self.pick_next_active_on_remove(available_paths)

        if emit:
            self.selection_changed.emit(self._selected_paths)

    def pick_next_active_on_remove(self, available_paths: list = None, emit: bool = True):
        """Promote a new active camera when current active is removed.

        Policy: pick first remaining selected; else pick first from available_paths; else clear.
        """
        new_active = None
        if self._selected_paths:
            # deterministic pick: sorted order
            new_active = sorted(self._selected_paths)[0]
        elif available_paths:
            new_active = available_paths[0] if len(available_paths) else None

        self._active_path = new_active
        if emit:
            self.active_changed.emit(self._active_path or "")
            self.selection_changed.emit(self._selected_paths)

    def select_range(self, start_path: str, end_path: str, ordered_paths: list, emit: bool = True):
        """Select a range between two paths given an ordering list (grid order).

        If either path is not found, fall back to selecting both if present.
        """
        if not ordered_paths:
            # fallback: just select both
            to_select = {p for p in (start_path, end_path) if p}
            self.set_selections(to_select, emit=emit)
            return

        try:
            i0 = ordered_paths.index(start_path)
            i1 = ordered_paths.index(end_path)
        except ValueError:
            to_select = {p for p in (start_path, end_path) if p}
            self.set_selections(to_select, emit=emit)
            return

        a, b = min(i0, i1), max(i0, i1)
        rng = ordered_paths[a:b+1]
        self.set_selections(rng, emit=emit)

    def serialize(self) -> list:
        """Return a serializable list of selected paths (active path first when present)."""
        out = []
        if self._active_path:
            out.append(self._active_path)
        out.extend([p for p in sorted(self._selected_paths) if p != self._active_path])
        return out

    def deserialize(self, paths_list: list, available_paths: list = None, emit: bool = True):
        """Restore selections from a list, optionally intersecting with available_paths."""
        if available_paths is not None:
            valid = [p for p in paths_list if p in available_paths]
        else:
            valid = list(paths_list)

        # First element may be active
        active = valid[0] if valid else None
        self._active_path = active
        self._selected_paths = set(valid)
        if self._active_path:
            self._selected_paths.add(self._active_path)
        if emit:
            self.active_changed.emit(self._active_path or "")
            self.selection_changed.emit(self._selected_paths)

    # Last-click helpers
    def set_last_clicked(self, path: str, emit: bool = True):
        self._last_clicked = path
        if emit:
            self.last_clicked_changed.emit(path or "")

    def get_last_clicked(self):
        return self._last_clicked
