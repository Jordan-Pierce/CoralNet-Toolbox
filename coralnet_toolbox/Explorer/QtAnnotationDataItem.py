import warnings

import os

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationDataItem:
    """Holds annotation information for consistent display across viewers."""

    def __init__(self, annotation, embedding_x=None, embedding_y=None, embedding_id=None):
        self.annotation = annotation
        self.embedding_x = embedding_x if embedding_x is not None else 0.0
        self.embedding_y = embedding_y if embedding_y is not None else 0.0
        self.embedding_id = embedding_id if embedding_id is not None else 0
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label
        self._marked_for_deletion = False  # Track deletion status

    @property
    def effective_label(self):
        """Get the current effective label (preview if exists, otherwise original)."""
        return self._preview_label if self._preview_label else self.annotation.label

    @property
    def effective_color(self):
        """Get the effective color for this annotation."""
        return self.effective_label.color

    @property
    def is_selected(self):
        """Check if this annotation is selected."""
        return self._is_selected

    def set_selected(self, selected):
        """Set the selection state."""
        self._is_selected = selected

    def set_preview_label(self, label):
        """Set a preview label for this annotation."""
        self._preview_label = label

    def clear_preview_label(self):
        """Clear the preview label and revert to original."""
        self._preview_label = None

    def has_preview_changes(self):
        """Check if this annotation has preview changes."""
        return self._preview_label is not None

    def mark_for_deletion(self):
        """Mark this annotation for deletion."""
        self._marked_for_deletion = True

    def unmark_for_deletion(self):
        """Unmark this annotation for deletion."""
        self._marked_for_deletion = False

    def is_marked_for_deletion(self):
        """Check if this annotation is marked for deletion."""
        return self._marked_for_deletion

    def apply_preview_permanently(self):
        """Apply the preview label permanently to the annotation."""
        if self._preview_label:
            self.annotation.update_label(self._preview_label)
            self.annotation.update_user_confidence(self._preview_label)
            self._original_label = self._preview_label
            self._preview_label = None
            return True
        return False

    def get_display_info(self):
        """Get display information for this annotation."""
        return {
            'id': self.annotation.id,
            'label': self.effective_label.short_label_code,
            'confidence': self.get_effective_confidence(),
            'type': type(self.annotation).__name__,
            'image': os.path.basename(self.annotation.image_path),
            'embedding_id': self.embedding_id,
            'color': self.effective_color
        }

    def get_effective_confidence(self):
        """Get the effective confidence value."""
        if self.annotation.verified and hasattr(self.annotation, 'user_confidence') and self.annotation.user_confidence:
            return list(self.annotation.user_confidence.values())[0]
        elif hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        return 0.0
