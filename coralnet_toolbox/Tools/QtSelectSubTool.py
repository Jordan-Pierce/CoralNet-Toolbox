from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsRectItem

from coralnet_toolbox.Tools.QtSubTool import SubTool


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SelectSubTool(SubTool):
    """SubTool for selecting multiple annotations with a rectangle."""
    
    def __init__(self, parent_tool):
        super().__init__(parent_tool)
        self.selection_rectangle = None
        self.selection_start_pos = None
        # Track marquee selection state while dragging
        self._marquee_current_ids: set = set()
        self._marquee_preexisting_ids: set = set()

    def activate(self, event, **kwargs):
        super().activate(event)
        self.selection_start_pos = self.annotation_window.mapToScene(event.pos())
        
        # Create and style the selection rectangle (dashed blue, light blue fill)
        self.selection_rectangle = QGraphicsRectItem()
        pen = QPen(QColor(0, 168, 230), 3, Qt.DashLine)
        pen.setCosmetic(True)
        self.selection_rectangle.setPen(pen)
        self.selection_rectangle.setBrush(QBrush(QColor(0, 168, 230, 30)))  # Light blue transparent fill
        self.selection_rectangle.setRect(QRectF(self.selection_start_pos, self.selection_start_pos))
        self.annotation_window.scene.addItem(self.selection_rectangle)
        # Capture the set of annotation ids that were selected before starting the marquee
        try:
            self._marquee_preexisting_ids = set(a.id for a in self.parent_tool.selected_annotations)
        except Exception:
            self._marquee_preexisting_ids = set()
        
    def deactivate(self):
        super().deactivate()
        if self.selection_rectangle:
            self.annotation_window.scene.removeItem(self.selection_rectangle)
            self.selection_rectangle = None
        self.selection_start_pos = None
        # Cleanup: deselect any annotations that were temporarily selected by the marquee
        if self._marquee_current_ids:
            # Build id -> annotation map
            id_map = {ann.id: ann for ann in self.annotation_window.get_image_annotations()}
            # Determine the set of ids that should remain selected after commit
            try:
                final_selected_ids = set(a.id for a in self.parent_tool.selected_annotations)
            except Exception:
                final_selected_ids = set()

            for ann_id in list(self._marquee_current_ids):
                # If this id was not selected before the marquee and is not in the final selection, deselect it
                if ann_id not in self._marquee_preexisting_ids and ann_id not in final_selected_ids:
                    ann = id_map.get(ann_id)
                    if ann:
                        ann.deselect()

        # Reset marquee tracking
        self._marquee_current_ids.clear()
        self._marquee_preexisting_ids.clear()

    def mouseMoveEvent(self, event):
        """Update the selection rectangle while dragging."""
        if not self.is_active or not self.selection_rectangle:
            return
            
        current_pos = self.annotation_window.mapToScene(event.pos())
        if self.annotation_window.cursorInWindow(event.pos()):
            rect = QRectF(self.selection_start_pos, current_pos).normalized()
            self.selection_rectangle.setRect(rect)
            # --- Live marquee selection: consider annotations inside rect as selected while dragging ---
            locked_label = self.parent_tool.get_locked_label()
            # Build id map and compute new set
            new_ids: set = set()
            id_map = {}
            for annotation in self.annotation_window.get_image_annotations():
                id_map[annotation.id] = annotation
                try:
                    center = annotation.center_xy
                except Exception:
                    center = None
                if center is None:
                    continue
                if rect.contains(center):
                    if locked_label and annotation.label.id != locked_label.id:
                        continue
                    new_ids.add(annotation.id)

            # Compute additions and removals relative to current marquee set
            additions = new_ids - self._marquee_current_ids
            removals = self._marquee_current_ids - new_ids

            # Select newly included annotations (but don't disturb those that were selected before drag)
            for ann_id in additions:
                if ann_id in self._marquee_preexisting_ids:
                    # already selected before marquee; leave it alone
                    continue
                ann = id_map.get(ann_id)
                if ann:
                    ann.select()

            # Deselect annotations that left the marquee (but keep those that were selected before drag)
            for ann_id in removals:
                if ann_id in self._marquee_preexisting_ids:
                    continue
                ann = id_map.get(ann_id)
                if ann:
                    ann.deselect()

            # Store new marquee set
            self._marquee_current_ids = new_ids

    def mouseReleaseEvent(self, event):
        """Finalize the selection and then deactivate."""
        self.finalize_selection()
        self.parent_tool.deactivate_subtool()

    def finalize_selection(self):
        """Select annotations contained within the drawn rectangle."""
        if not self.selection_rectangle:
            return

        rect = self.selection_rectangle.rect()
        locked_label = self.parent_tool.get_locked_label()

        # Iterate through all annotations to check for inclusion
        for annotation in self.annotation_window.get_image_annotations():
            if rect.contains(annotation.center_xy):
                if locked_label and annotation.label.id != locked_label.id:
                    continue  # Skip if label is locked and doesn't match
                if annotation not in self.parent_tool.selected_annotations:
                    # Append to selection (multi-select is the default for marquee)
                    self.annotation_window.select_annotation(annotation, multi_select=True)