"""Central data store for all annotations across all images.

This module provides the AnnotationManager class which owns the annotation
dictionaries and action stack previously held inside AnnotationWindow.
By extracting these into a separate QObject, any view (AnnotationWindow,
context canvases, explorer windows) can query annotation data and subscribe
to change signals without routing through the monolithic UI class.
"""

import warnings

from PyQt5.QtCore import QObject, pyqtSignal

from coralnet_toolbox.QtActions import ActionStack

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationManager(QObject):
    """Central data store for all annotations across all images.

    Owns the core data structures:
        - annotations_dict: {uuid: Annotation} — all annotations
        - image_annotations_dict: {image_path: [Annotation]} — per-image index
        - selected_annotations: list — currently selected annotations
        - action_stack: ActionStack — undo/redo

    Signals are relayed from AnnotationWindow so external consumers
    (context canvases, explorer windows) can subscribe here instead.
    """

    # Relay signals (bridged from AnnotationWindow signals)
    annotationAdded = pyqtSignal(str)          # annotation_id
    annotationsAdded = pyqtSignal(list)        # list of annotation_ids
    annotationRemoved = pyqtSignal(str)        # annotation_id
    annotationsRemoved = pyqtSignal(list)      # list of annotation_ids
    annotationModified = pyqtSignal(str)       # annotation_id
    annotationLabelChanged = pyqtSignal(str, str)  # annotation_id, new_label
    selectionChanged = pyqtSignal(object)      # list of annotation IDs

    def __init__(self, parent=None):
        """Initialize the annotation manager."""
        super().__init__(parent)

        # Core data stores
        self.annotations_dict = {}            # {uuid: Annotation}
        self.image_annotations_dict = {}      # {image_path: [Annotation]}
        self.selected_annotations = []        # Currently selected annotations
        self.action_stack = ActionStack()      # Undo/Redo

    def get_image_annotations(self, image_path):
        """Get all annotations for the given image path.

        Args:
            image_path (str): Path to the image.

        Returns:
            list: List of Annotation objects for the image, or empty list.
        """
        return self.image_annotations_dict.get(image_path, [])
