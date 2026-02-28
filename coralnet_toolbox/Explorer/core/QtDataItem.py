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
    
        if self.viewer and self.viewer.display_mode == 'sprites':
            ar = self.data_item.aspect_ratio
            if ar >= 1.0:
                width = SPRITE_SIZE * scale_factor
                height = (SPRITE_SIZE / ar) * scale_factor
            else:
                height = SPRITE_SIZE * scale_factor
                width = (SPRITE_SIZE * ar) * scale_factor
            return QRectF(0, 0, width, height)
        else:
            
            size = POINT_SIZE * scale_factor
            return QRectF(0, 0, size, size)

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
        effective_label = self.data_item.effective_label
        base_color = self.data_item.effective_color
        
        display_color = base_color
        dash_color = base_color

        if effective_label and effective_label.id == "-1":
            display_color = QColor("black")
        
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
                pen_width = min(2, int(2 * scale_factor))
                half_w = pen_width / 2.0
                rect = self.boundingRect().adjusted(half_w, half_w, -half_w, -half_w)
                
                # Draw background border
                bg_pen = QPen(display_color.darker(160), pen_width + 2)
                bg_pen.setJoinStyle(Qt.MiterJoin)
                painter.setPen(bg_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
                
                # Draw animated dashed stroke
                fg_pen = QPen(dash_color, pen_width)
                fg_pen.setJoinStyle(Qt.MiterJoin)
                fg_pen.setDashPattern([4.0, 4.0])
                fg_pen.setDashOffset(self.animation_offset)
                painter.setPen(fg_pen)
                painter.drawRect(rect)
            else:
                faint = QColor(display_color.darker(160))
                faint.setAlpha(80)
                painter.setPen(QPen(faint, 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(self.boundingRect())

        else:
            # --- MODERN DATA-SCIENCE DOTS ---
            if self.isSelected():
                painter.setPen(Qt.NoPen)
                painter.setBrush(display_color)
                painter.drawEllipse(self.boundingRect())
                
                pen_width = max(1.5, 2.0 * scale_factor)
                half_w = pen_width / 2.0
                draw_rect = self.boundingRect().adjusted(half_w, half_w, -half_w, -half_w)
                
                white_pen = QPen(display_color.darker(160), pen_width + 1)
                painter.setPen(white_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(draw_rect)
                
                dash_pen = QPen(dash_color, pen_width)
                dash_pen.setDashPattern([2.0, 2.0])
                dash_pen.setDashOffset(self.animation_offset)
                painter.setPen(dash_pen)
                painter.drawEllipse(draw_rect)
                
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
                self.animate()
            else:      # It is being deselected
                self.deanimate()
                
        elif change == QGraphicsItem.ItemSceneChange and value is None:
            # Clean up if it gets deleted from the scene
            self.deanimate()
            
        return super().itemChange(change, value)
    
    def tick_animation(self):
        """Perform one 'tick' of the marching ants animation."""
        self.animation_offset = (self.animation_offset + 1) % 8
        self.update()  # Trigger repaint
        
    def animate(self):
        """Register with the global animation timer."""
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)
            
    def deanimate(self):
        """Unregister from the global animation timer and reset."""
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
            
        self.animation_offset = 0  # Reset the dash offset
        try:
            self.update()  # Trigger a final repaint to draw the solid line
        except RuntimeError:
            pass
    
    def __del__(self):
        """Clean up the timer when the item is deleted."""
        if hasattr(self, 'is_animating') and self.is_animating:
            self.deanimate()
            

class AnnotationImageWidget(QWidget):
    """Widget to display a single annotation image crop with selection support."""

    def __init__(self, data_item, widget_height=96, annotation_viewer=None, parent=None):
        super(AnnotationImageWidget, self).__init__(parent)
        self.data_item = data_item
        self.annotation = data_item.annotation
        self.annotation_viewer = annotation_viewer

        self.widget_height = widget_height
        self.aspect_ratio = 1.0
        self.pixmap = None
        self.is_loaded = False

        # --- High Performance Rendering Caches ---
        self._cached_scaled_pixmap = None
        self._cached_size = None
        self._cached_label_id = None
        self._cached_tag_text = ""
        self._cached_text_width = 0
        self._cached_text_height = 0
        self._font = QFont("Arial", 6, QFont.Bold)

        # --- Animation Properties ---
        self.animation_manager = None
        self.is_animating = False
        self.animation_offset = 0

        self.recalculate_aspect_ratio()
        self.update_height(self.widget_height)
        self.update_tooltip()
        
    def set_animation_manager(self, manager):
        self.animation_manager = manager
        
    def is_graphics_item_valid(self):
        """
        Checks if the widget is still valid.
        """
        try:
            # Just verify the underlying C++ object still exists.
            # Do NOT check self.isVisible() here, because the virtualized 
            # scroll area temporarily hides widgets when they go off-screen!
            self.objectName()
            return True
        except RuntimeError:
            return False
        
    def update_tooltip(self):
        self.setToolTip(self.data_item.get_tooltip_text())

    def recalculate_aspect_ratio(self):
        annotation = self.data_item.annotation
        if hasattr(annotation, 'cropped_bbox'):
            min_x, min_y, max_x, max_y = annotation.cropped_bbox
            width = max_x - min_x
            height = max_y - min_y
            if height > 0:
                self.aspect_ratio = width / height
                return
        
        try:
            top_left = annotation.get_bounding_box_top_left()
            bottom_right = annotation.get_bounding_box_bottom_right()
            if top_left and bottom_right:
                width = bottom_right.x() - top_left.x()
                height = bottom_right.y() - top_left.y()
                if height > 0:
                    self.aspect_ratio = width / height
                    return
        except (AttributeError, TypeError):
            pass
        
        try:
            pixmap = annotation.get_cropped_image()
            if pixmap and not pixmap.isNull() and pixmap.height() > 0:
                self.aspect_ratio = pixmap.width() / pixmap.height()
                return
        except (AttributeError, TypeError):
            pass
        
        self.aspect_ratio = 1.0

    def load_image(self):
        if self.is_loaded:
            return
        try:
            cropped_pixmap = self.annotation.get_cropped_image_graphic()
            if cropped_pixmap and not cropped_pixmap.isNull():
                self.pixmap = cropped_pixmap
                self.is_loaded = True
                self._cached_scaled_pixmap = None  # Invalidate cache
                self.update()
            else:
                self.pixmap = None
                self.update()
        except Exception as e:
            print(f"Error loading annotation image: {e}")
            self.pixmap = None
            self.update()

    def unload_image(self):
        if not self.is_loaded:
            return
        self.pixmap = None
        self._cached_scaled_pixmap = None  # Clean up memory
        self.is_loaded = False

    def update_height(self, new_height):
        self.widget_height = new_height
        new_width = int(self.widget_height * self.aspect_ratio)
        self.setFixedSize(new_width, new_height)
        self._cached_scaled_pixmap = None  # Invalidate cache
        self.update()

    def update_selection_visuals(self):
        if self.data_item.is_selected:
            self.animate()
        else:
            self.deanimate()
        self.update()

    def is_selected(self):
        return self.data_item.is_selected

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw the image (Centered & Cached)
        if self.pixmap:
            current_size = self.size()
            # ONLY execute smooth transformation if the widget size has changed
            if self._cached_scaled_pixmap is None or self._cached_size != current_size:
                self._cached_scaled_pixmap = self.pixmap.scaled(
                    self.width() - 8,
                    self.height() - 8,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self._cached_size = current_size
                
            x_offset = (self.width() - self._cached_scaled_pixmap.width()) // 2
            y_offset = (self.height() - self._cached_scaled_pixmap.height()) // 2
            painter.drawPixmap(x_offset, y_offset, self._cached_scaled_pixmap)
        else:
            painter.setPen(QColor("black"))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image\nAvailable")

        effective_label = self.data_item.effective_label
        pen_color = self.data_item.effective_color
        dashed_color = QColor(pen_color)  # Base color copy

        if effective_label and effective_label.id == "-1":
            pen_color = QColor("black")

        half_width = (ANNOTATION_WIDTH - 1) // 2
        rect = self.rect().adjusted(half_width, half_width, -half_width, -half_width)

        # 2a. Draw the Contrast Stroke
        bg_pen = QPen(pen_color.darker(160), ANNOTATION_WIDTH + 2)
        bg_pen.setCosmetic(True)
        bg_pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(bg_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        # 2b. Draw the Colored Line
        if self.is_selected():
            pen = QPen(dashed_color, ANNOTATION_WIDTH)
            pen.setDashPattern([4.0, 4.0])
            pen.setDashOffset(self.animation_offset)
            pen.setJoinStyle(Qt.MiterJoin)
        else:
            pen = QPen(pen_color, ANNOTATION_WIDTH)
            pen.setStyle(Qt.SolidLine)
            pen.setJoinStyle(Qt.MiterJoin)

        painter.setPen(pen)
        painter.drawRect(rect)

        # 3. Draw the Floating Nametag (Cached Font Metrics)
        if self._cached_label_id != effective_label.id:
            # Re-calculate expensive OS string layout ONLY when the label changes
            from PyQt5.QtGui import QFontMetrics
            self._cached_label_id = effective_label.id
            self._cached_tag_text = effective_label.short_label_code
            fm = QFontMetrics(self._font)
            self._cached_text_width = fm.horizontalAdvance(self._cached_tag_text)
            self._cached_text_height = fm.height()

        painter.setFont(self._font)
        
        pad_x, pad_y = 4, 2
        bg_rect = QRectF(4, 4, self._cached_text_width + pad_x * 2, self._cached_text_height + pad_y * 2)
        
        bg_color = QColor(pen_color)
        bg_color.setAlpha(255) 
        painter.setPen(QPen(pen_color.darker(160), 1)) 
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(bg_rect, 4, 4)
        
        # Smart text contrast
        luminance = (0.299 * bg_color.red() + 0.587 * bg_color.green() + 0.114 * bg_color.blue()) / 255
        text_color = bg_color.darker(200) if luminance > 0.5 else bg_color.lighter(200)
    
        painter.setPen(text_color)
        painter.drawText(bg_rect, Qt.AlignCenter, self._cached_tag_text)
        
    def tick_animation(self):
        """Perform one 'tick' of the marching ants animation."""
        # Check visibility here instead! This saves CPU cycles by pausing 
        # the math and repaint requests while the widget is scrolled out of view.
        if not self.isVisible():
            return 
            
        self.animation_offset = (self.animation_offset + 1) % 8
        self.update()  # Trigger repaint
        
    def animate(self):
        self.is_animating = True
        if self.animation_manager:
            self.animation_manager.register_animating_object(self)
            
    def deanimate(self):
        self.is_animating = False
        if self.animation_manager:
            self.animation_manager.unregister_animating_object(self)
        self.animation_offset = 0
        try:
            self.update()
        except RuntimeError:
            pass
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_selection'):
                self.annotation_viewer.handle_annotation_selection(self, event)
                event.accept()
                return
        elif event.button() == Qt.RightButton:
            if self.annotation_viewer and hasattr(self.annotation_viewer, 'handle_annotation_context_menu'):
                self.annotation_viewer.handle_annotation_context_menu(self, event)
                event.accept()
                return
            else:
                event.ignore()
        super().mousePressEvent(event)
        
    def showEvent(self, event):
        """Ensure animation resumes smoothly when scrolled back into view."""
        super().showEvent(event)
        if self.is_selected():
            self.animate()
        
    def __del__(self):
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
