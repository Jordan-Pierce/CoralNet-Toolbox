# coralnet_toolbox/Explorer/QtEmbeddingPointModel.py
"""
Lightweight data model for the Embedding Viewer scatter plot.

This module provides a minimal wrapper around Annotation objects specifically
for use in the embedding visualization. It contains X/Y/Z coordinates, quality
scores, and anomaly detection results - no gallery layout properties.
"""

import os
import warnings

from PyQt5.QtGui import QColor

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPointModel:
    """
    Lightweight wrapper for embedding visualization state.
    
    This class holds only the state needed for displaying an annotation in the
    embedding scatter plot. It references the core Annotation object for ID
    and label, but manages its own coordinate space and ML metrics.
    
    Attributes:
        annotation: Reference to the core Annotation object.
        embedding_x, embedding_y, embedding_z: Current display coordinates.
        embedding_x_3d, embedding_y_3d, embedding_z_3d: Original 3D coordinates.
        embedding_id: Index in the embedding array.
        _is_selected: Whether this point is currently selected.
        anomaly_score: Anomaly detection score (0-1).
        quality_score: Composite quality metric (0-1).
        local_density: Local density in feature space.
        spatial_consistency: Consistency with nearby annotations.
    """
    
    __slots__ = [
        'annotation', 'embedding_x', 'embedding_y', 'embedding_z',
        'embedding_x_3d', 'embedding_y_3d', 'embedding_z_3d',
        'embedding_id', '_is_selected', 'anomaly_score', 'quality_score',
        'local_density', 'spatial_consistency', '_preview_label'
    ]
    
    def __init__(self, annotation, embedding_x=None, embedding_y=None, embedding_id=None):
        """
        Initialize the embedding point model.
        
        Args:
            annotation: The core Annotation object to wrap.
            embedding_x: Initial X coordinate (default 0.0).
            embedding_y: Initial Y coordinate (default 0.0).
            embedding_id: Index in the embedding array.
        """
        self.annotation = annotation
        
        # Display coordinates (may be rotated)
        self.embedding_x = embedding_x if embedding_x is not None else 0.0
        self.embedding_y = embedding_y if embedding_y is not None else 0.0
        self.embedding_z = 0.0
        
        # Original 3D coordinates from dimensionality reduction
        self.embedding_x_3d = 0.0
        self.embedding_y_3d = 0.0
        self.embedding_z_3d = 0.0
        
        self.embedding_id = embedding_id
        self._is_selected = False
        self._preview_label = None
        
        # ML quality and anomaly metrics
        self.anomaly_score = None
        self.quality_score = None
        self.local_density = None
        self.spatial_consistency = None
    
    @property
    def effective_label(self):
        """Get the current effective label."""
        return self._preview_label if self._preview_label else self.annotation.label
    
    @property
    def effective_color(self):
        """Get the effective color for this point based on the label."""
        return self.effective_label.color
    
    @property
    def is_selected(self):
        """Check if this point is selected."""
        return self._is_selected
    
    def set_selected(self, selected):
        """Set the selection state."""
        self._is_selected = selected
    
    def set_preview_label(self, label):
        """Set a preview label."""
        self._preview_label = label
    
    def clear_preview_label(self):
        """Clear the preview label."""
        self._preview_label = None
    
    def has_preview_changes(self):
        """Check if this point has a preview label."""
        return self._preview_label is not None
    
    def set_coordinates(self, x, y, z=0.0):
        """
        Set the display coordinates.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate (default 0.0).
        """
        self.embedding_x = x
        self.embedding_y = y
        self.embedding_z = z
    
    def set_3d_coordinates(self, x, y, z):
        """
        Set the original 3D coordinates from dimensionality reduction.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        self.embedding_x_3d = x
        self.embedding_y_3d = y
        self.embedding_z_3d = z
        
        # Also update display coordinates initially
        self.embedding_x = x
        self.embedding_y = y
        self.embedding_z = z
    
    def get_effective_confidence(self):
        """Get the effective confidence value."""
        # External machine_confidence
        if hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        
        # Verified status
        if self.annotation.verified:
            return 1.0
        
        return 1.0
    
    def calculate_quality_score(self):
        """
        Calculate a composite quality score (0-1).
        
        Factors:
        - Model confidence
        - Local density
        - Spatial consistency
        - Annotation geometry
        """
        weights = {
            'confidence': 0.4,
            'density': 0.3,
            'spatial': 0.2,
            'geometry': 0.1
        }
        
        scores = {}
        
        # Confidence score
        scores['confidence'] = self.get_effective_confidence()
        
        # Local density score
        if self.local_density is not None:
            scores['density'] = min(1.0, self.local_density / 10.0)
        else:
            scores['density'] = 0.5
        
        # Spatial consistency
        if self.spatial_consistency is not None:
            scores['spatial'] = self.spatial_consistency
        else:
            scores['spatial'] = 0.5
        
        # Geometry score
        geometry_score = 1.0
        try:
            top_left = self.annotation.get_bounding_box_top_left()
            bottom_right = self.annotation.get_bounding_box_bottom_right()
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                
                # Penalize very small annotations
                if width < 5 or height < 5:
                    geometry_score *= 0.7
                
                # Penalize very large annotations
                if width > 1000 or height > 1000:
                    geometry_score *= 0.8
        except (AttributeError, TypeError):
            pass
        
        scores['geometry'] = geometry_score
        
        # Weighted average
        total_weight = sum(weights.values())
        self.quality_score = sum(scores[k] * weights[k] for k in scores) / total_weight
        return self.quality_score
    
    def get_quality_info(self):
        """Get formatted quality information for display."""
        if self.quality_score is None:
            return ""
        
        quality_percent = int(self.quality_score * 100)
        
        if quality_percent >= 80:
            color, rating = "green", "Excellent"
        elif quality_percent >= 60:
            color, rating = "lightgreen", "Good"
        elif quality_percent >= 40:
            color, rating = "orange", "Fair"
        else:
            color, rating = "red", "Poor"
        
        info_parts = [f"<b style='color: {color}'>Quality: {quality_percent}% ({rating})</b>"]
        
        if self.local_density is not None:
            info_parts.append(f"Density: {self.local_density:.2f}")
        
        if self.spatial_consistency is not None:
            info_parts.append(f"Spatial Consistency: {int(self.spatial_consistency * 100)}%")
        
        if self.anomaly_score is not None:
            info_parts.append(f"Anomaly Score: {int(self.anomaly_score * 100)}%")
        
        return "<br>".join(info_parts)
    
    def get_anomaly_info(self):
        """Get formatted anomaly information for display."""
        if self.anomaly_score is None:
            return ""
        
        anomaly_percent = int(self.anomaly_score * 100)
        
        if anomaly_percent >= 80:
            color, rating = "red", "Very Anomalous"
        elif anomaly_percent >= 60:
            color, rating = "orange", "Anomalous"
        elif anomaly_percent >= 40:
            color, rating = "gold", "Slightly Anomalous"
        else:
            color, rating = "green", "Normal"
        
        return f"<b style='color: {color}'>Anomaly: {anomaly_percent}% ({rating})</b>"
    
    def get_display_info(self):
        """Get display information for this point."""
        return {
            'id': self.annotation.id,
            'label': self.effective_label.short_label_code,
            'confidence': self.get_effective_confidence(),
            'type': type(self.annotation).__name__,
            'image': os.path.basename(self.annotation.image_path),
            'embedding_id': self.embedding_id,
            'color': self.effective_color,
            'x': self.embedding_x,
            'y': self.embedding_y,
            'z': self.embedding_z,
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
        
        # Add quality info
        quality_info = self.get_quality_info()
        if quality_info:
            tooltip_parts.append(f"<hr>{quality_info}")
        
        # Add anomaly info
        anomaly_info = self.get_anomaly_info()
        if anomaly_info:
            tooltip_parts.append(anomaly_info)
        
        return "<br>".join(tooltip_parts)
