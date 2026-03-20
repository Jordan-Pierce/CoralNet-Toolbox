import warnings

import os

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter, QFont, QBrush
from PyQt5.QtWidgets import QGraphicsObject, QStyle, QWidget, QGraphicsItem

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


POINT_SIZE = 15
POINT_WIDTH = 2
SPRITE_SIZE = 48

ANNOTATION_WIDTH = 4


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingPointItem(QGraphicsObject):
    """
    A custom QGraphicsObject that can display as a dot or an image sprite,
    getting its state from an associated AnnotationDataItem.
    """

    def __init__(self, data_item, viewer):
        """
        Initializes the point item.
        Args:
            data_item (AnnotationDataItem): The data item that holds the state.
            viewer (EmbeddingViewer): A reference to the parent viewer.
        """
        super(EmbeddingPointItem, self).__init__()

        self.data_item = data_item
        self.viewer = viewer
        self.thumbnail_pixmap = None

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        # Use the annotation's label color (darkened) for outlines instead of pure black
        try:
            dark_outline = QColor(self.data_item.effective_color).darker(160)
        except Exception:
            dark_outline = QColor("black")
        self.default_pen = QPen(dark_outline, POINT_WIDTH)
        self.default_pen.setCosmetic(True)
        self.setPos(self.data_item.embedding_x, self.data_item.embedding_y)
        self.setToolTip(self.data_item.get_tooltip_text())
        
        # --- Animation Properties ---
        self.animation_manager = None
        self.is_animating = False
        
        # --- Marching Ants offset ---
        self.animation_offset = 0
        
    def set_animation_manager(self, manager):
        """
        Binds this object to the central AnimationManager.
        
        Args:
            manager (AnimationManager): The central animation manager instance.
        """
        # Keep a reference but do NOT register with the global manager.
        # These items draw a static selection outline; animation ticks
        # are intentionally disabled to avoid per-frame timers.
        self.animation_manager = manager
        
    def is_graphics_item_valid(self):
        """
        Checks if the graphics item is still valid and added to a scene.
        
        Returns:
            bool: True if the item exists and has a scene, False otherwise.
        """
        try:
            return self.scene() is not None
        except RuntimeError:
            # This can happen if the C++ part of the item is deleted
            return False

    def boundingRect(self):
        """Returns the bounding rectangle, which depends on the display mode and depth."""
        
        scale_factor = 1.0
        if self.viewer and self.viewer.is_3d_data and self.viewer.z_range > 0:
            # Normalize z from its global range to a [0, 1] range
            z_normalized = (self.data_item.embedding_z - self.viewer.min_z) / self.viewer.z_range
            # Map normalized z to a scale factor (e.g., from 0.5x to 1.5x)
            scale_factor = 0.5 + z_normalized
    
        # Allow viewer to override base sizes (dynamic resizing via Ctrl+Wheel)
        base_sprite = getattr(self.viewer, 'sprite_size', SPRITE_SIZE) if self.viewer else SPRITE_SIZE
        base_point = getattr(self.viewer, 'point_size', POINT_SIZE) if self.viewer else POINT_SIZE

        if self.viewer and self.viewer.display_mode == 'sprites':
            ar = self.data_item.aspect_ratio
            if ar >= 1.0:
                width = base_sprite * scale_factor
                height = (base_sprite / ar) * scale_factor
            else:
                height = base_sprite * scale_factor
                width = (base_sprite * ar) * scale_factor
            # Center the rect at the origin
            return QRectF(-width / 2, -height / 2, width, height)
        else:
            size = base_point * scale_factor
            # Center the rect at the origin
            return QRectF(-size / 2, -size / 2, size, size)

    def update_tooltip(self):
        """Updates the tooltip by fetching the latest text from the data item."""
        self.setToolTip(self.data_item.get_tooltip_text())

    def paint(self, painter, option, widget):
        """Clean, high-performance data-science aesthetic."""
        option.state &= ~QStyle.State_Selected
        painter.setRenderHint(QPainter.Antialiasing)
        
        scale_factor = 1.0
        opacity = 255
        if self.viewer and self.viewer.is_3d_data and self.viewer.z_range > 0:
            z_normalized = (self.data_item.embedding_z - self.viewer.min_z) / self.viewer.z_range
            scale_factor = 0.5 + z_normalized
            opacity = int(128 + 127 * z_normalized)

        # Directly grab the existing effective color reference (Avoids wrapping memory unnecessarily)
        display_color = self.data_item.effective_color
        dash_color = self.data_item.effective_color
        
        display_mode = self.viewer.display_mode if self.viewer else 'dots'

        if display_mode == 'sprites':
            current_size = self.boundingRect().size().toSize()
            if self.thumbnail_pixmap is None or self.thumbnail_pixmap.size() != current_size:
                source_pixmap = self.data_item.annotation.get_cropped_image_graphic()
                if source_pixmap and not source_pixmap.isNull():
                    self.thumbnail_pixmap = source_pixmap.scaled(
                        current_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
            
            if self.thumbnail_pixmap:
                painter.drawPixmap(self.boundingRect().topLeft(), self.thumbnail_pixmap)

            if self.isSelected():
                pen_width = ANNOTATION_WIDTH
                buffer = 4.0
                halo_rect = self.boundingRect().adjusted(-buffer, -buffer, buffer, buffer)
                
                # Draw dashed halo with buffer
                halo_pen = QPen(dash_color, pen_width)
                halo_pen.setStyle(Qt.DashLine)
                halo_pen.setCosmetic(True)
                painter.setPen(halo_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(halo_rect)
            else:
                faint = QColor(display_color.darker(160))
                faint.setAlpha(80)
                faint_pen = QPen(faint, 1)
                faint_pen.setCosmetic(True)
                painter.setPen(faint_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(self.boundingRect())

        else:
            # --- MODERN DATA-SCIENCE DOTS ---
            if self.isSelected():
                painter.setPen(Qt.NoPen)
                painter.setBrush(display_color)
                painter.drawEllipse(self.boundingRect())
                
                # Draw concentric dashed halo with buffer
                pen_width = max(1.5, 2.0 * scale_factor)
                buffer = 4.0
                point_radius = self.boundingRect().width() / 2.0
                halo_radius = point_radius + buffer
                center = self.boundingRect().center()
                halo_rect = QRectF(center.x() - halo_radius, center.y() - halo_radius, 
                                   halo_radius * 2, halo_radius * 2)
                
                halo_pen = QPen(display_color, pen_width)
                halo_pen.setStyle(Qt.DashLine)
                halo_pen.setCosmetic(True)
                painter.setPen(halo_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(halo_rect)
                
            else:
                effective_brush_color = QColor(display_color)
                effective_brush_color.setAlpha(opacity)
                painter.setPen(Qt.NoPen)
                painter.setBrush(effective_brush_color)
                painter.drawEllipse(self.boundingRect())
    
    def itemChange(self, change, value):
        """Safely handle selection state changes without spamming the paint loop."""
        if change == QGraphicsItem.ItemSelectedChange:
            if value:  # It is being selected
                # Bring this item to the front so it is not occluded by other points.
                # O(1) assignment instead of an O(N) scene iteration
                self.setZValue(1000)
                self.animate()
            else:      # It is being deselected
                # Reset z-value so normal stacking resumes
                self.setZValue(0)
                self.deanimate()
                
        elif change == QGraphicsItem.ItemSceneChange and value is None:
            # Clean up if it gets deleted from the scene
            self.deanimate()
            
        return super().itemChange(change, value)
    
    def tick_animation(self):
        """Perform one 'tick' of the marching ants animation."""
        # Animations disabled for gallery/embedding points to keep
        # rendering static and avoid global timer registration.
        return
        
    def animate(self):
        """Enable animated state visually without registering global timers.

        We keep the flag and trigger a repaint so selection visuals update,
        but do not register with the global AnimationManager.
        """
        self.is_animating = True
        try:
            self.update()
        except RuntimeError:
            pass
            
    def deanimate(self):
        """Disable animated state and reset visuals (no global unregister)."""
        self.is_animating = False
        self.animation_offset = 0
        try:
            self.update()
        except RuntimeError:
            pass
    
    def __del__(self):
        """Clean up the timer when the item is deleted."""
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()


class AnnotationDataItem:
    """
    Holds all annotation state information for consistent display across viewers.
    This acts as the "ViewModel" for a single annotation, serving as the single
    source of truth for its state in the UI.
    """

    def __init__(self, annotation, embedding_x=None, embedding_y=None, embedding_id=None):
        self.annotation = annotation
        
        self.embedding_x = embedding_x if embedding_x is not None else 0.0
        self.embedding_y = embedding_y if embedding_y is not None else 0.0
        
        self.embedding_z = 0.0  # This will store the rotated Z-value (depth)
        
        # Store the original, un-rotated 3D coordinates from the embedding
        self.embedding_x_3d = 0.0
        self.embedding_y_3d = 0.0
        self.embedding_z_3d = 0.0
        
        self.embedding_id = embedding_id
        
        self._is_selected = False
        self._preview_label = None
        self._original_label = annotation.label

        # Calculate and store aspect ratio on initialization
        self.aspect_ratio = self._calculate_aspect_ratio()
        
        # sklearn predictions from Auto-Annotation Wizard (session-only, temporary)
        # self.sklearn_prediction = None  # Stores sklearn model predictions during Explorer session

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
        """Get the current effective label (preview if it exists, otherwise original)."""
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
        """Set the selection state. This is the single point of control."""
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
    
    def get_tooltip_text(self):
        """
        Generates a rich HTML-formatted tooltip with all relevant information.
        """
        info = self.get_display_info()
        
        tooltip_parts = [
            f"<b>ID:</b> {info['id']}",
            f"<b>Image:</b> {info['image']}",
            f"<b>Label:</b> {info['label']}",
            f"<b>Type:</b> {info['type']}"
        ]

        # Add sklearn prediction details if they exist (from Auto-Annotation Wizard)
        # if hasattr(self, 'sklearn_prediction') and self.sklearn_prediction:
        #     pred = self.sklearn_prediction
        #     if 'top_predictions' in pred:
        #         pred_parts = ["<b>Model Predictions:</b>"]
        #         for p in pred['top_predictions'][:3]:  # Top 3
        #             pred_parts.append(f"{p['label']}: {p['confidence']:.1%}")
        #         tooltip_parts.append(f"<hr>{'<br>'.join(pred_parts)}")

        return "<br>".join(tooltip_parts)

    def get_effective_confidence(self):
        """
        Get the effective confidence value with proper priority:
        1. sklearn predictions (from Auto-Annotation Wizard, session-only)
        2. External machine_confidence (from MainWindow, permanent)
        3. Verified status (1.0 if verified)
        4. Default (1.0)
        
        Returns:
            float: Confidence value between 0 and 1
        """
        # Priority 1: sklearn predictions from Auto-Annotation Wizard (session-only)
        # if hasattr(self, 'sklearn_prediction') and self.sklearn_prediction is not None:
        #     if isinstance(self.sklearn_prediction, dict) and 'confidence' in self.sklearn_prediction:
        #         return float(self.sklearn_prediction['confidence'])
        
        # Priority 2: External machine_confidence (from CoralNet or other tools)
        if hasattr(self.annotation, 'machine_confidence') and self.annotation.machine_confidence:
            return list(self.annotation.machine_confidence.values())[0]
        
        # Priority 3: Verified status
        if self.annotation.verified:
            return 1.0
            
        # Default
        return 1.0
