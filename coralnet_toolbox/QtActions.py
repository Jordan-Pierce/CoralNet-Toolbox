"""Action classes for undo/redo operations used by AnnotationWindow.

This module houses lightweight Action implementations that operate on
an AnnotationWindow instance passed into action constructors. Keeping
these in a separate module avoids cluttering the large
`QtAnnotationWindow.py` file and prevents circular imports.
"""

from typing import Any
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import QApplication


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Action:
    """Base class for undo/redo actions."""

    def do(self) -> Any:
        raise NotImplementedError

    def undo(self) -> Any:
        raise NotImplementedError

    def merge_with(self, other: "Action") -> bool:
        """Attempt to merge another action into this one.

        Return True if merge succeeded (other absorbed into self) and False
        otherwise. Default implementation does not merge.
        """
        return False


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
    """Add multiple annotations (undo by deleting them in bulk)."""

    def __init__(self, annotation_window, annotations):
        self.annotation_window = annotation_window
        self.annotations = annotations

    def do(self):
        self.annotation_window.add_annotations(self.annotations, record_action=False)
        try:
            current = self.annotation_window.current_image_path
            if current:
                image_paths = {getattr(a, 'image_path', None) for a in self.annotations}
                if current in image_paths:
                    self.annotation_window.load_annotations(image_path=current)
                    try:
                        self.annotation_window.scene.update()
                        self.annotation_window.viewport().update()
                    except Exception:
                        pass
        except Exception:
            pass

    def undo(self):
        # FIX: Use true bulk delete instead of a massive loop
        self.annotation_window.delete_annotations(self.annotations, record_action=False)


class DeleteAnnotationsAction(Action):
    """Delete multiple annotations (undo by re-adding them in bulk)."""

    def __init__(self, annotation_window, annotations):
        self.annotation_window = annotation_window
        self.annotations = annotations

    def do(self):
        # FIX: Use true bulk delete instead of a massive loop
        self.annotation_window.delete_annotations(self.annotations, record_action=False)

    def undo(self):
        # FIX: Use true bulk add instead of a massive loop
        self.annotation_window.add_annotations(self.annotations, record_action=False)


class MoveAnnotationAction(Action):
    """Move an annotation (undo by moving back to old center)."""

    def __init__(self, annotation_window, annotation_id, old_center, new_center):
        self.annotation_window = annotation_window
        self.annotation_id = annotation_id
        self.old_center = old_center
        self.new_center = new_center

    def do(self):
        try:
            self.annotation_window.set_annotation_location(self.annotation_id, self.new_center)
        except Exception:
            pass

    def undo(self):
        try:
            self.annotation_window.set_annotation_location(self.annotation_id, self.old_center)
        except Exception:
            pass


class ResizeAnnotationAction(Action):
    """Resize an annotation (undo by restoring old size/scale)."""

    def __init__(self, annotation_window, annotation_id, old_size, new_size):
        self.annotation_window = annotation_window
        self.annotation_id = annotation_id
        self.old_size = old_size
        self.new_size = new_size

    def _apply_size(self, annotation, size_value):
        try:
            # Mirror the logic used elsewhere: temporarily disconnect heavy slots
            try:
                
                annotation.annotationUpdated.disconnect(
                    self.annotation_window.main_window.confidence_window.display_cropped_image
                )
                annotation.annotationUpdated.disconnect(self.annotation_window.on_annotation_updated)
            except Exception:
                pass

            annotation.update_annotation_size(size_value)
            # Refresh cropped image display if possible
            try:
                annotation.create_cropped_image(self.annotation_window.rasterio_image)
                self.annotation_window.main_window.confidence_window.display_cropped_image(annotation)
            except Exception:
                pass

            try:
                annotation.annotationUpdated.connect(
                    self.annotation_window.main_window.confidence_window.display_cropped_image
                )
                annotation.annotationUpdated.connect(self.annotation_window.on_annotation_updated)
            except Exception:
                pass
        except Exception:
            pass

    def do(self):
        annotation = self.annotation_window.annotations_dict.get(self.annotation_id)
        if annotation:
            self._apply_size(annotation, self.new_size)

    def undo(self):
        annotation = self.annotation_window.annotations_dict.get(self.annotation_id)
        if annotation:
            self._apply_size(annotation, self.old_size)

    def merge_with(self, other: "Action") -> bool:
        # Merge consecutive resize actions for the same annotation by keeping
        # the original old_size and updating new_size to the incoming action's
        # new_size.
        try:
            if not isinstance(other, ResizeAnnotationAction):
                return False
            if other.annotation_id != self.annotation_id:
                return False
            # Adopt the other's new_size as the merged new_size
            self.new_size = other.new_size
            return True
        except Exception:
            return False


class ChangeLabelAction(Action):
    """Change label on a single annotation (undo restores old label)."""

    def __init__(self, annotation_window, annotation_id, old_label, new_label):
        self.annotation_window = annotation_window
        self.annotation_id = annotation_id
        self.old_label = old_label
        self.new_label = new_label

    def do(self):
        annotation = self.annotation_window.annotations_dict.get(self.annotation_id)
        if annotation:
            try:
                annotation.update_user_confidence(self.new_label)
                annotation.create_cropped_image(self.annotation_window.rasterio_image)
                try:
                    self.annotation_window.main_window.confidence_window.display_cropped_image(annotation)
                except Exception:
                    pass
            except Exception:
                pass

    def undo(self):
        annotation = self.annotation_window.annotations_dict.get(self.annotation_id)
        if annotation:
            try:
                annotation.update_user_confidence(self.old_label)
                annotation.create_cropped_image(self.annotation_window.rasterio_image)
                try:
                    self.annotation_window.main_window.confidence_window.display_cropped_image(annotation)
                except Exception:
                    pass
            except Exception:
                pass


class ChangeLabelsAction(Action):
    """Apply label changes to multiple annotations (undo restores previous labels)."""

    def __init__(self, annotation_window, changes_list):
        # changes_list: list of (annotation_id, old_label, new_label)
        self.annotation_window = annotation_window
        self.changes_list = changes_list

    def do(self):
        for ann_id, old_label, new_label in self.changes_list:
            ann = self.annotation_window.annotations_dict.get(ann_id)
            if ann:
                try:
                    ann.update_user_confidence(new_label)
                except Exception:
                    pass

    def undo(self):
        for ann_id, old_label, new_label in self.changes_list:
            ann = self.annotation_window.annotations_dict.get(ann_id)
            if ann:
                try:
                    ann.update_user_confidence(old_label)
                except Exception:
                    pass


class CutAnnotationAction(Action):
    """Cut/split an annotation into new annotations (undo restores original)."""

    def __init__(self, annotation_window, original_annotation, new_annotations):
        self.annotation_window = annotation_window
        self.original_annotation = original_annotation
        self.new_annotations = new_annotations

    def do(self):
        try:
            # Delete original then add new ones (suppress recording)
            self.annotation_window.delete_annotation(self.original_annotation.id, record_action=False)
            self.annotation_window.add_annotations(self.new_annotations, record_action=False)
        except Exception:
            pass

    def undo(self):
        try:
            # Remove new ones then re-add original
            for ann in self.new_annotations:
                self.annotation_window.delete_annotation(ann.id, record_action=False)
            self.annotation_window.add_annotation(self.original_annotation, record_action=False)
        except Exception:
            pass


class MergeAnnotationsAction(Action):
    """Merge multiple annotations into one (undo restores originals)."""

    def __init__(self, annotation_window, original_annotations, merged_annotation):
        self.annotation_window = annotation_window
        self.original_annotations = original_annotations
        self.merged_annotation = merged_annotation

    def do(self):
        try:
            # Delete originals and add merged
            self.annotation_window.delete_annotations(self.original_annotations)
            self.annotation_window.add_annotation(self.merged_annotation, record_action=False)
        except Exception:
            pass

    def undo(self):
        try:
            # Remove merged and restore originals
            self.annotation_window.delete_annotation(self.merged_annotation.id, record_action=False)
            self.annotation_window.add_annotations(self.original_annotations, record_action=False)
        except Exception:
            pass


class SplitAnnotationAction(Action):
    """Split a multi-part annotation into multiple annotations (undo restores original)."""

    def __init__(self, annotation_window, original_annotation, split_annotations):
        self.annotation_window = annotation_window
        self.original_annotation = original_annotation
        self.split_annotations = split_annotations

    def do(self):
        try:
            self.annotation_window.delete_annotation(self.original_annotation.id, record_action=False)
            self.annotation_window.add_annotations(self.split_annotations, record_action=False)
        except Exception:
            pass

    def undo(self):
        try:
            for ann in self.split_annotations:
                self.annotation_window.delete_annotation(ann.id, record_action=False)
            self.annotation_window.add_annotation(self.original_annotation, record_action=False)
        except Exception:
            pass


class MoveImageAnnotationsAction(Action):
    """Move a set of annotations for an image (undo restores original centers).

    Provide movements as a list of (annotation_id, old_center, new_center).
    """

    def __init__(self, annotation_window, movements):
        self.annotation_window = annotation_window
        # movements: list of (annotation_id, old_center, new_center)
        self.movements = movements

    def do(self):
        for ann_id, old_c, new_c in self.movements:
            try:
                self.annotation_window.set_annotation_location(ann_id, new_c)
            except Exception:
                pass

    def undo(self):
        for ann_id, old_c, new_c in self.movements:
            try:
                self.annotation_window.set_annotation_location(ann_id, old_c)
            except Exception:
                pass


class AnnotationGeometryEditAction(Action):
    """Generic geometry edit (store old/new geometry and apply)."""
    def __init__(self, annotation_window, annotation_id, old_geom, new_geom):
        self.annotation_window = annotation_window
        self.annotation_id = annotation_id
        # Store geometries in a lightweight serialized form to avoid holding
        # QObject/QPointF-heavy structures on the stack.
        self.old_geom = self._serialize_geom(old_geom)
        self.new_geom = self._serialize_geom(new_geom)

    def _serialize_geom(self, geom):
        """Serialize geometry into lightweight tuples.

        Returns ('poly', outer_pts, holes) or ('rect', tl, br).
        outer_pts: list of (x,y), holes: list of lists of (x,y)
        """
        try:
            # If already serialized, return as-is
            if isinstance(geom, tuple) and geom and geom[0] in ('poly', 'rect'):
                return geom
        except Exception:
            pass

        # Polygon-like (pts, holes) or pts list
        try:
            if isinstance(geom, tuple) and len(geom) == 2:
                pts, holes = geom
                outer = []
                if pts:
                    for p in pts:
                        try:
                            outer.append((float(p.x()), float(p.y())))
                        except Exception:
                            outer.append((float(p[0]), float(p[1])))
                hs = []
                if holes:
                    for hole in holes:
                        hl = []
                        for p in hole:
                            try:
                                hl.append((float(p.x()), float(p.y())))
                            except Exception:
                                hl.append((float(p[0]), float(p[1])))
                        hs.append(hl)
                return ('poly', outer, hs)
        except Exception:
            pass

        try:
            if isinstance(geom, list):
                outer = []
                for p in geom:
                    try:
                        outer.append((float(p.x()), float(p.y())))
                    except Exception:
                        outer.append((float(p[0]), float(p[1])))
                return ('poly', outer, [])
        except Exception:
            pass

        # Rectangle-like
        try:
            if isinstance(geom, tuple) and len(geom) == 2:
                a, b = geom
                try:
                    return ('rect', (float(a.x()), float(a.y())), (float(b.x()), float(b.y())))
                except Exception:
                    return ('rect', (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        except Exception:
            pass

        return ('poly', [], [])

    def _apply_geom(self, geom_serialized):
        ann = self.annotation_window.annotations_dict.get(self.annotation_id)
        if not ann:
            return
        try:
            kind = geom_serialized[0]
            if kind == 'poly' and hasattr(ann, 'points'):
                outer = geom_serialized[1] if len(geom_serialized) > 1 else []
                holes = geom_serialized[2] if len(geom_serialized) > 2 else []

                ann.points = [QPointF(x, y) for x, y in outer]
                if hasattr(ann, 'holes'):
                    ann.holes = [[QPointF(x, y) for x, y in hole] for hole in holes]
                try:
                    ann.set_centroid()
                    ann.set_cropped_bbox()
                except Exception:
                    pass
                ann.update_graphics_item()
                ann.annotationUpdated.emit(ann)
                try:
                    ann.create_cropped_image(self.annotation_window.rasterio_image)
                    self.annotation_window.main_window.confidence_window.display_cropped_image(ann)
                except Exception:
                    pass
                
                # Emit signals to notify viewers of geometry change
                self.annotation_window.annotationGeometryEdited.emit(
                    self.annotation_id,
                    {'old_geom': self.old_geom, 'new_geom': self.new_geom}
                )
                self.annotation_window.annotationModified.emit(self.annotation_id)
                return

            if kind == 'rect' and not hasattr(ann, 'points'):
                tl = geom_serialized[1]
                br = geom_serialized[2]
                ann.top_left = QPointF(tl[0], tl[1])
                ann.bottom_right = QPointF(br[0], br[1])
                try:
                    ann.set_centroid()
                    ann.set_cropped_bbox()
                except Exception:
                    pass
                ann.update_graphics_item()
                ann.annotationUpdated.emit(ann)
                try:
                    ann.create_cropped_image(self.annotation_window.rasterio_image)
                    self.annotation_window.main_window.confidence_window.display_cropped_image(ann)
                except Exception:
                    pass
                
                # Emit signals to notify viewers of geometry change
                self.annotation_window.annotationGeometryEdited.emit(
                    self.annotation_id,
                    {'old_geom': self.old_geom, 'new_geom': self.new_geom}
                )
                self.annotation_window.annotationModified.emit(self.annotation_id)
                return

            # Fallback: try update_polygon with original form
            try:
                ann.update_polygon(geom_serialized)
                # Even for fallback, notify viewers
                self.annotation_window.annotationGeometryEdited.emit(
                    self.annotation_id,
                    {'old_geom': self.old_geom, 'new_geom': self.new_geom}
                )
                self.annotation_window.annotationModified.emit(self.annotation_id)
            except Exception:
                pass
        except Exception:
            pass

    def do(self):
        self._apply_geom(self.new_geom)

    def undo(self):
        self._apply_geom(self.old_geom)

    def merge_with(self, other: "Action") -> bool:
        # Merge geometry edits on the same annotation by keeping the original
        # old_geom and updating new_geom to the incoming action's new_geom.
        try:
            if not isinstance(other, AnnotationGeometryEditAction):
                return False
            if other.annotation_id != self.annotation_id:
                return False
            self.new_geom = other.new_geom
            return True
        except Exception:
            return False


class SubtractAnnotationsAction(Action):
    """Subtract (cookie-cutter) operation: replace originals with result annotations."""

    def __init__(self, annotation_window, original_annotations, result_annotations):
        self.annotation_window = annotation_window
        self.original_annotations = original_annotations
        self.result_annotations = result_annotations

    def do(self):
        try:
            # Add results then delete originals
            for ann in self.result_annotations:
                self.annotation_window.add_annotation(ann, record_action=False)
            for ann in self.original_annotations:
                self.annotation_window.delete_annotation(ann.id, record_action=False)
        except Exception:
            pass

    def undo(self):
        try:
            # Remove results then re-add originals
            for ann in self.result_annotations:
                self.annotation_window.delete_annotation(ann.id, record_action=False)
            for ann in self.original_annotations:
                self.annotation_window.add_annotation(ann, record_action=False)
        except Exception:
            pass


class ActionStack:
    """Simple undo/redo stack for Action objects."""

    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    def push(self, action: Action):
        # Show busy cursor while manipulating stacks to keep UI responsive
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Try to merge with the last undo action if possible
            if self.undo_stack:
                last = None
                try:
                    last = self.undo_stack[-1]
                except Exception:
                    last = None

                try:
                    if last and last.merge_with(action):
                        # Merged into last action; clear redo and do not push
                        self.redo_stack.clear()
                        return
                except Exception:
                    # If merge check fails, fall back to normal push
                    pass

            self.undo_stack.append(action)
            self.redo_stack.clear()
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def undo(self):
        # Use a busy cursor while performing undo to indicate work
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            if self.undo_stack:
                action = self.undo_stack.pop()
                action.undo()
                self.redo_stack.append(action)
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass

    def redo(self):
        # Use a busy cursor while performing redo to indicate work
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            if self.redo_stack:
                action = self.redo_stack.pop()
                action.do()
                self.undo_stack.append(action)
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass


__all__ = [
    "Action",
    "AddAnnotationAction",
    "DeleteAnnotationAction",
    "ActionStack",
    "AddAnnotationsAction",
    "DeleteAnnotationsAction",
    "MoveAnnotationAction",
    "ResizeAnnotationAction",
    "ChangeLabelAction",
    "ChangeLabelsAction",
    "CutAnnotationAction",
    "MergeAnnotationsAction",
    "SplitAnnotationAction",
    "MoveImageAnnotationsAction",
    "SubtractAnnotationsAction",
    "AnnotationGeometryEditAction",
]

