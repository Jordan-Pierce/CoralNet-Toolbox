"""Action classes for undo/redo operations used by AnnotationWindow.

This module houses lightweight Action implementations that operate on
an AnnotationWindow instance passed into action constructors. Keeping
these in a separate module avoids cluttering the large
`QtAnnotationWindow.py` file and prevents circular imports.
"""

from typing import Any


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Action:
    """Base class for undo/redo actions."""

    def do(self) -> Any:
        raise NotImplementedError

    def undo(self) -> Any:
        raise NotImplementedError


class AddAnnotationAction(Action):
    """Add an annotation (undo by deleting it).

    The action holds a reference to the annotation and the annotation
    window instance. Methods operate via the public API on the
    annotation_window to avoid importing the AnnotationWindow class
    directly.
    """

    def __init__(self, annotation_window, annotation):
        self.annotation_window = annotation_window
        self.annotation = annotation

    def do(self):
        # Only add if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.add_annotation_from_tool(self.annotation, record_action=False)

    def undo(self):
        # Only delete if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.delete_annotation(self.annotation.id, record_action=False)


class DeleteAnnotationAction(Action):
    """Delete an annotation (undo by re-adding it)."""

    def __init__(self, annotation_window, annotation):
        self.annotation_window = annotation_window
        self.annotation = annotation

    def do(self):
        # Only delete if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.delete_annotation(self.annotation.id, record_action=False)

    def undo(self):
        # Only add if the annotation's image is the current image
        if self.annotation_window.current_image_path == self.annotation.image_path:
            self.annotation_window.add_annotation_from_tool(self.annotation, record_action=False)


class AddAnnotationsAction(Action):
    """Add multiple annotations (undo by deleting them)."""

    def __init__(self, annotation_window, annotations):
        self.annotation_window = annotation_window
        self.annotations = annotations

    def do(self):
        # Use the bulk add method but suppress recording another bulk action
        # (the ActionStack already contains this action)
        self.annotation_window.add_annotations(self.annotations, record_action=False)

        # If any of the added annotations belong to the currently displayed image,
        # (re)load annotations for that image so graphics items are created.
        try:
            current = self.annotation_window.current_image_path
            if current:
                image_paths = {getattr(a, 'image_path', None) for a in self.annotations}
                if current in image_paths:
                    # load_annotations will create graphics for visible labels
                    self.annotation_window.load_annotations(image_path=current)
                    # Ensure view refresh
                    try:
                        self.annotation_window.scene.update()
                        self.annotation_window.viewport().update()
                    except Exception:
                        pass
        except Exception:
            # Non-fatal — best-effort to refresh UI
            pass

    def undo(self):
        # Delete each annotation without recording individual undo actions
        for annotation in self.annotations:
            self.annotation_window.delete_annotation(annotation.id, record_action=False)


class DeleteAnnotationsAction(Action):
    """Delete multiple annotations (undo by re-adding them)."""

    def __init__(self, annotation_window, annotations):
        self.annotation_window = annotation_window
        self.annotations = annotations

    def do(self):
        # Delete each annotation without creating separate undo actions
        for annotation in self.annotations:
            self.annotation_window.delete_annotation(annotation.id, record_action=False)

    def undo(self):
        # Re-add each annotation without recording actions
        for annotation in self.annotations:
            self.annotation_window.add_annotation(annotation, record_action=False)


class ActionStack:
    """Simple undo/redo stack for Action objects."""

    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push(self, action: Action):
        self.undo_stack.append(action)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            action.undo()
            self.redo_stack.append(action)

    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            action.do()
            self.undo_stack.append(action)


__all__ = [
    "Action",
    "AddAnnotationAction",
    "DeleteAnnotationAction",
    "ActionStack",
    "AddAnnotationsAction",
    "DeleteAnnotationsAction",
]

