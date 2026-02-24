# coralnet_toolbox/Explorer/QtGalleryItemModel.py
"""
Lightweight data model for the Annotation Gallery viewer.

This module provides a minimal wrapper around Annotation objects specifically
for use in the gallery display. It only contains properties needed for grid
layout, sorting, and visual state - no embedding coordinates or ML metrics.
"""

import os
import warnings

from PyQt5.QtGui import QColor

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class GalleryItemModel:
    """
    Lightweight wrapper for gallery display state.
    
    This class holds only the state needed for displaying an annotation in the
    gallery grid. It references the core Annotation object for ID, label, and
    image path, but does not allocate memory for embedding coordinates or
    ML quality metrics.
    
    Attributes:
        annotation: Reference to the core Annotation object.
        aspect_ratio: Calculated width/height ratio for layout.
        _is_selected: Whether this item is currently selected.
        _preview_label: Temporary preview label (for bulk labeling).
        _thumbnail_loaded: Whether the thumbnail image has been loaded.
    """
    
    __slots__ = [
        'annotation', 'aspect_ratio', '_is_selected', '_preview_label',
        '_original_label', '_thumbnail_loaded', 'sklearn_prediction'
    ]
    
    def __init__(self, annotation):
        """
        Initialize the gallery item model.
        
        Args:
            annotation: The core Annotation object to wrap.
        """
        self.annotation = annotation
        self.aspect_ratio = self._calculate_aspect_ratio()
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label
        self._thumbnail_loaded = False
        
        # sklearn predictions from Auto-Annotation Wizard (session-only)
        self.sklearn_prediction = None
    
    def _calculate_aspect_ratio(self):
        """Calculate and return the annotation's aspect ratio."""
        annotation = self.annotation
        
        if hasattr(annotation, 'cropped_bbox'):
            min_x, min_y, max_x, max_y = annotation.cropped_bbox
            width = max_x - min_x
            height = max_y - min_y
            if height > 0:
                return width / height

        try:
            top_left = annotation.get_bounding_box_top_left()
            bottom_right = annotation.get_bounding_box_bottom_right()
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                if height > 0:
                    return width / height
        except (AttributeError, TypeError):
            pass

        try:
            pixmap = annotation.get_cropped_image()
            if pixmap and not pixmap.isNull() and pixmap.height() > 0:
                return pixmap.width() / pixmap.height()
        except (AttributeError, TypeError):
            pass
        
        return 1.0  # Default to square
    
    @property
    def effective_label(self):
        """Get the current effective label (preview if exists, otherwise original)."""
        return self._preview_label if self._preview_label else self.annotation.label
    
    @property
    def effective_color(self):
        """Get the effective color for this annotation based on the effective label."""
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
        """Clear the preview label and revert to the original."""
        self._preview_label = None
    
    def has_preview_changes(self):
        """Check if this annotation has a temporary preview label assigned."""
        return self._preview_label is not None
    
    def apply_preview_permanently(self):
        """Apply the preview label permanently to the underlying annotation object."""
        if self._preview_label:
            self.annotation.update_label(self._preview_label)
            self.annotation.update_user_confidence(self._preview_label)
            self._original_label = self._preview_label
            self._preview_label = None
            return True
        return False
    
    def get_effective_confidence(self):
        """
        Get the effective confidence value.
        
        Priority:
        1. sklearn predictions (from Auto-Annotation Wizard)
        2. External machine_confidence
        3. Verified status (1.0 if verified)
        4. Default (1.0)
        """
        # Priority 1: sklearn predictions
        if self.sklearn_prediction is not None:
            if isinstance(self.sklearn_prediction, dict) and 'confidence' in self.sklearn_prediction:
                return float(self.sklearn_prediction['confidence'])
        
        # Priority 2: External machine_confidence
        if hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        
        # Priority 3: Verified status
        if self.annotation.verified:
            return 1.0
        
        # Default
        return 1.0
    
    def get_confidence_color(self, breaks=None):
        """
        Get a color for the confidence score.
        
        Args:
            breaks: List of confidence break points for quantile-based coloring.
        
        Returns:
            QColor: Color representing the confidence level.
        """
        confidence = self.get_effective_confidence()
        
        if confidence == 0:
            return QColor(128, 128, 128)  # Gray
        
        if breaks is None or len(breaks) < 2:
            # Default 6-tier system
            if confidence <= 0.17:
                return QColor(220, 20, 60)   # Crimson
            elif confidence <= 0.33:
                return QColor(255, 99, 71)   # Tomato
            elif confidence <= 0.50:
                return QColor(255, 165, 0)   # Orange
            elif confidence <= 0.67:
                return QColor(255, 215, 0)   # Gold
            elif confidence <= 0.83:
                return QColor(144, 238, 144) # Light green
            else:
                return QColor(34, 139, 34)   # Dark green
        else:
            # Quantile breaks
            if len(breaks) >= 5:
                if confidence <= breaks[0]:
                    return QColor(220, 20, 60)
                elif confidence <= breaks[1]:
                    return QColor(255, 99, 71)
                elif confidence <= breaks[2]:
                    return QColor(255, 165, 0)
                elif confidence <= breaks[3]:
                    return QColor(255, 215, 0)
                elif confidence <= breaks[4]:
                    return QColor(144, 238, 144)
                else:
                    return QColor(34, 139, 34)
            else:
                if confidence <= breaks[0]:
                    return QColor(220, 20, 60)
                else:
                    return QColor(34, 139, 34)
    
    def get_display_info(self):
        """Get display information for this annotation."""
        return {
            'id': self.annotation.id,
            'label': self.effective_label.short_label_code,
            'confidence': self.get_effective_confidence(),
            'type': type(self.annotation).__name__,
            'image': os.path.basename(self.annotation.image_path),
            'color': self.effective_color
        }
    
    def get_tooltip_text(self):
        """Generate a tooltip with relevant information."""
        info = self.get_display_info()
        
        tooltip_parts = [
            f"<b>ID:</b> {info['id']}",
            f"<b>Image:</b> {info['image']}",
            f"<b>Label:</b> {info['label']}",
            f"<b>Type:</b> {info['type']}"
        ]
        
        # Add sklearn prediction details if available
        if self.sklearn_prediction:
            pred = self.sklearn_prediction
            if 'top_predictions' in pred:
                pred_parts = ["<b>Model Predictions:</b>"]
                for p in pred['top_predictions'][:3]:
                    pred_parts.append(f"{p['label']}: {p['confidence']:.1%}")
                tooltip_parts.append(f"<hr>{'<br>'.join(pred_parts)}")
        
        return "<br>".join(tooltip_parts)
