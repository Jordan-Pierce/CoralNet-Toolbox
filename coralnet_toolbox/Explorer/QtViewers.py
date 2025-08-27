import warnings

import os

from PyQt5.QtGui import QPen, QColor, QPainter, QBrush, QPainterPath, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QPointF, pyqtSignal, QSignalBlocker, pyqtSlot, QEvent
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QGraphicsView, QScrollArea,
                             QGraphicsScene, QPushButton, QComboBox, QLabel, QWidget,
                             QSlider, QMessageBox, QGraphicsRectItem, QRubberBand, QMenu,
                             QWidgetAction, QToolButton, QAction)

from coralnet_toolbox.Explorer.QtDataItem import EmbeddingPointItem
from coralnet_toolbox.Explorer.QtDataItem import AnnotationImageWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import SimilaritySettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import UncertaintySettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import MislabelSettingsWidget
from coralnet_toolbox.Explorer.QtSettingsWidgets import DuplicateSettingsWidget

from coralnet_toolbox.Annotations.QtRectangleAnnotation import RectangleAnnotation

from coralnet_toolbox.Icons import get_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

POINT_WIDTH = 3

# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class EmbeddingViewer(QWidget):
    """Custom QGraphicsView for interactive embedding visualization with an isolate mode."""
    selection_changed = pyqtSignal(list)
    reset_view_requested = pyqtSignal()
    find_mislabels_requested = pyqtSignal()
    mislabel_parameters_changed = pyqtSignal(dict) 
    find_uncertain_requested = pyqtSignal()
    uncertainty_parameters_changed = pyqtSignal(dict)
    find_duplicates_requested = pyqtSignal()
    duplicate_parameters_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        """Initialize the EmbeddingViewer widget."""
        super(EmbeddingViewer, self).__init__(parent)
        self.explorer_window = parent

        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)

        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMinimumHeight(200)

        self.rubber_band = None
        self.rubber_band_origin = QPointF()
        self.selection_at_press = None
        self.points_by_id = {}
        self.previous_selection_ids = set()

        # State for isolate mode
        self.isolated_mode = False
        self.isolated_points = set()
        
        self.is_uncertainty_analysis_available = False

        self.animation_offset = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)

        # New timer for virtualization
        self.view_update_timer = QTimer(self)
        self.view_update_timer.setSingleShot(True)
        self.view_update_timer.timeout.connect(self._update_visible_points)

        self.graphics_scene.selectionChanged.connect(self.on_selection_changed)
        self.setup_ui()
        self.graphics_view.mousePressEvent = self.mousePressEvent
        self.graphics_view.mouseDoubleClickEvent = self.mouseDoubleClickEvent
        self.graphics_view.mouseReleaseEvent = self.mouseReleaseEvent
        self.graphics_view.mouseMoveEvent = self.mouseMoveEvent
        self.graphics_view.wheelEvent = self.wheelEvent

    def setup_ui(self):
        """Set up the UI with toolbar layout and graphics view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar_layout = QHBoxLayout()

        # Isolate/Show All buttons
        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setToolTip("Hide all non-selected points")
        self.isolate_button.clicked.connect(self.isolate_selection)
        toolbar_layout.addWidget(self.isolate_button)

        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setToolTip("Show all embedding points")
        self.show_all_button.clicked.connect(self.show_all_points)
        toolbar_layout.addWidget(self.show_all_button)
        
        toolbar_layout.addWidget(self._create_separator())
                
        # Create a QToolButton to have both a primary action and a dropdown menu
        self.find_mislabels_button = QToolButton()
        self.find_mislabels_button.setText("Find Potential Mislabels")
        self.find_mislabels_button.setPopupMode(QToolButton.MenuButtonPopup)  # Key change for split-button style
        self.find_mislabels_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.find_mislabels_button.setStyleSheet(
            "QToolButton::menu-indicator {"
            " subcontrol-position: right center;"
            " subcontrol-origin: padding;"
            " left: -4px;"
            " }"
        )

        # The primary action (clicking the button) triggers the analysis
        run_analysis_action = QAction("Find Potential Mislabels", self)
        run_analysis_action.triggered.connect(self.find_mislabels_requested.emit)
        self.find_mislabels_button.setDefaultAction(run_analysis_action)

        # The dropdown menu contains the settings
        mislabel_settings_widget = MislabelSettingsWidget()
        settings_menu = QMenu(self)
        widget_action = QWidgetAction(settings_menu)
        widget_action.setDefaultWidget(mislabel_settings_widget)
        settings_menu.addAction(widget_action)
        self.find_mislabels_button.setMenu(settings_menu)
        
        # Connect the widget's signal to the viewer's signal
        mislabel_settings_widget.parameters_changed.connect(self.mislabel_parameters_changed.emit)
        toolbar_layout.addWidget(self.find_mislabels_button)
        
        # Create a QToolButton for uncertainty analysis
        self.find_uncertain_button = QToolButton()
        self.find_uncertain_button.setText("Review Uncertain")
        self.find_uncertain_button.setToolTip(
            "Find annotations where the model is least confident.\n"
            "Requires a .pt classification model and 'Predictions' mode."
        )
        self.find_uncertain_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.find_uncertain_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.find_uncertain_button.setStyleSheet(
            "QToolButton::menu-indicator { "
            "subcontrol-position: right center; "
            "subcontrol-origin: padding; "
            "left: -4px; }"
        )
        
        run_uncertainty_action = QAction("Review Uncertain", self)
        run_uncertainty_action.triggered.connect(self.find_uncertain_requested.emit)
        self.find_uncertain_button.setDefaultAction(run_uncertainty_action)

        uncertainty_settings_widget = UncertaintySettingsWidget()
        uncertainty_menu = QMenu(self)
        uncertainty_widget_action = QWidgetAction(uncertainty_menu)
        uncertainty_widget_action.setDefaultWidget(uncertainty_settings_widget)
        uncertainty_menu.addAction(uncertainty_widget_action)
        self.find_uncertain_button.setMenu(uncertainty_menu)
        
        uncertainty_settings_widget.parameters_changed.connect(self.uncertainty_parameters_changed.emit)
        toolbar_layout.addWidget(self.find_uncertain_button)
        
        # Create a QToolButton for duplicate detection
        self.find_duplicates_button = QToolButton()
        self.find_duplicates_button.setText("Find Duplicates")
        self.find_duplicates_button.setToolTip(
            "Find annotations that are likely duplicates based on feature similarity."
        )
        self.find_duplicates_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.find_duplicates_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.find_duplicates_button.setStyleSheet(
            "QToolButton::menu-indicator { "
            "subcontrol-position: right center; "
            "subcontrol-origin: padding; "
            "left: -4px; }"
        )

        run_duplicates_action = QAction("Find Duplicates", self)
        run_duplicates_action.triggered.connect(self.find_duplicates_requested.emit)
        self.find_duplicates_button.setDefaultAction(run_duplicates_action)

        duplicate_settings_widget = DuplicateSettingsWidget()
        duplicate_menu = QMenu(self)
        duplicate_widget_action = QWidgetAction(duplicate_menu)
        duplicate_widget_action.setDefaultWidget(duplicate_settings_widget)
        duplicate_menu.addAction(duplicate_widget_action)
        self.find_duplicates_button.setMenu(duplicate_menu)
        
        duplicate_settings_widget.parameters_changed.connect(self.duplicate_parameters_changed.emit)
        toolbar_layout.addWidget(self.find_duplicates_button)
    
        # Add a stretch and separator
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self._create_separator())

        # Center on selection button
        self.center_on_selection_button = QPushButton()
        self.center_on_selection_button.setIcon(get_icon("target.png"))
        self.center_on_selection_button.setToolTip("Center view on selected point(s)")
        self.center_on_selection_button.clicked.connect(self.center_on_selection)
        toolbar_layout.addWidget(self.center_on_selection_button)
                
        # Home button to reset view
        self.home_button = QPushButton()
        self.home_button.setIcon(get_icon("home.png"))
        self.home_button.setToolTip("Reset view to fit all points")
        self.home_button.clicked.connect(self.reset_view)
        toolbar_layout.addWidget(self.home_button)
        
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.graphics_view)

        self.placeholder_label = QLabel(
            "No embedding data available.\nPress 'Apply Embedding' to generate visualization."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(self.placeholder_label)

        self.show_placeholder()
        self._update_toolbar_state()
        
    def _create_separator(self):
        """Creates a vertical separator for the toolbar."""
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; margin: 0 5px;")
        return separator
        
    def _schedule_view_update(self):
        """Schedules a delayed update of visible points to avoid performance issues."""
        self.view_update_timer.start(50)  # 50ms delay

    def _update_visible_points(self):
        """Sets visibility for points based on whether they are in the viewport."""
        if self.isolated_mode or not self.points_by_id:
            return

        # Get the visible rectangle in scene coordinates
        visible_rect = self.graphics_view.mapToScene(self.graphics_view.viewport().rect()).boundingRect()
        
        # Add a buffer to make scrolling smoother by loading points before they enter the view
        buffer_x = visible_rect.width() * 0.2
        buffer_y = visible_rect.height() * 0.2
        buffered_visible_rect = visible_rect.adjusted(-buffer_x, -buffer_y, buffer_x, buffer_y)

        for point in self.points_by_id.values():
            point.setVisible(buffered_visible_rect.contains(point.pos()) or point.isSelected())

    @pyqtSlot()
    def isolate_selection(self):
        """Hides all points that are not currently selected."""
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items or self.isolated_mode:
            return

        self.isolated_points = set(selected_items)
        self.graphics_view.setUpdatesEnabled(False)
        try:
            for point in self.points_by_id.values():
                point.setVisible(point in self.isolated_points)
            self.isolated_mode = True
        finally:
            self.graphics_view.setUpdatesEnabled(True)

        self._update_toolbar_state()

    @pyqtSlot()
    def show_all_points(self):
        """Shows all embedding points, exiting isolated mode."""
        if not self.isolated_mode:
            return

        self.isolated_mode = False
        self.isolated_points.clear()
        self.graphics_view.setUpdatesEnabled(False)
        try:
            # Instead of showing all, let the virtualization logic take over
            self._update_visible_points()
        finally:
            self.graphics_view.setUpdatesEnabled(True)

        self._update_toolbar_state()

    def _update_toolbar_state(self):
        """Updates toolbar buttons based on selection and isolation mode."""
        selection_exists = bool(self.graphics_scene.selectedItems())
        points_exist = bool(self.points_by_id)

        self.find_mislabels_button.setEnabled(points_exist)
        self.find_uncertain_button.setEnabled(points_exist and self.is_uncertainty_analysis_available)
        self.find_duplicates_button.setEnabled(points_exist)
        self.center_on_selection_button.setEnabled(points_exist and selection_exists)

        if self.isolated_mode:
            self.isolate_button.hide()
            self.show_all_button.show()
        else:
            self.isolate_button.show()
            self.show_all_button.hide()
            self.isolate_button.setEnabled(selection_exists)

    def reset_view(self):
        """Reset the view to fit all embedding points."""
        self.fit_view_to_points()
        
    def center_on_selection(self):
        """Centers the view on selected point(s) or maintains the current view if no points are selected."""
        selected_items = self.graphics_scene.selectedItems()
        if not selected_items:
            # No selection, show a message
            QMessageBox.information(self, "No Selection", "Please select one or more points first.")
            return
            
        # Create a bounding rect that encompasses all selected points
        selection_rect = None
        
        for item in selected_items:
            if isinstance(item, EmbeddingPointItem):
                # Get the item's bounding rect in scene coordinates
                item_rect = item.sceneBoundingRect()
                
                # Add padding around the point for better visibility
                padding = 50  # pixels
                item_rect = item_rect.adjusted(-padding, -padding, padding, padding)
                
                if selection_rect is None:
                    selection_rect = item_rect
                else:
                    selection_rect = selection_rect.united(item_rect)
        
        if selection_rect:
            # Add extra margin for better visibility
            margin = 20
            selection_rect = selection_rect.adjusted(-margin, -margin, margin, margin)
            
            # Fit the view to the selection rect
            self.graphics_view.fitInView(selection_rect, Qt.KeepAspectRatio)

    def show_placeholder(self):
        """Show the placeholder message and hide the graphics view."""
        self.graphics_view.setVisible(False)
        self.placeholder_label.setVisible(True)
        self.home_button.setEnabled(False)
        self.center_on_selection_button.setEnabled(False)  # Disable center button
        self.find_mislabels_button.setEnabled(False)
        self.find_uncertain_button.setEnabled(False)
        self.find_duplicates_button.setEnabled(False)

        self.isolate_button.show()
        self.isolate_button.setEnabled(False)
        self.show_all_button.hide()

    def show_embedding(self):
        """Show the graphics view and hide the placeholder message."""
        self.graphics_view.setVisible(True)
        self.placeholder_label.setVisible(False)
        self.home_button.setEnabled(True)
        self._update_toolbar_state()

    # Delegate graphics view methods
    def setRenderHint(self, hint):
        """Set render hint for the graphics view."""
        self.graphics_view.setRenderHint(hint)

    def setDragMode(self, mode):
        """Set drag mode for the graphics view."""
        self.graphics_view.setDragMode(mode)

    def setTransformationAnchor(self, anchor):
        """Set transformation anchor for the graphics view."""
        self.graphics_view.setTransformationAnchor(anchor)

    def setResizeAnchor(self, anchor):
        """Set resize anchor for the graphics view."""
        self.graphics_view.setResizeAnchor(anchor)

    def mapToScene(self, point):
        """Map a point to the scene coordinates."""
        return self.graphics_view.mapToScene(point)

    def scale(self, sx, sy):
        """Scale the graphics view."""
        self.graphics_view.scale(sx, sy)

    def translate(self, dx, dy):
        """Translate the graphics view."""
        self.graphics_view.translate(dx, dy)

    def fitInView(self, rect, aspect_ratio):
        """Fit the view to a rectangle with aspect ratio."""
        self.graphics_view.fitInView(rect, aspect_ratio)

    def keyPressEvent(self, event):
        """Handles key presses for deleting selected points."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and event.modifiers() == Qt.ControlModifier:
            selected_items = self.graphics_scene.selectedItems()
            if not selected_items:
                super().keyPressEvent(event)
                return

            # Extract the central data items from the selected graphics points
            data_items_to_delete = [
                item.data_item for item in selected_items if isinstance(item, EmbeddingPointItem)
            ]

            # Delegate the actual deletion to the main ExplorerWindow
            if data_items_to_delete:
                self.explorer_window.delete_data_items(data_items_to_delete)

            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for selection (point or rubber band) and panning."""
        # Ctrl+Right-Click for context menu selection 
        if event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                # 1. Clear all selections in both viewers
                self.graphics_scene.clearSelection()
                item_at_pos.setSelected(True)
                self.on_selection_changed()  # Updates internal state and emits signals

                # 2. Sync annotation viewer selection
                ann_id = item_at_pos.data_item.annotation.id
                self.explorer_window.annotation_viewer.render_selection_from_ids({ann_id})

                # 3. Update annotation window (set image, select, center)
                explorer = self.explorer_window
                annotation = item_at_pos.data_item.annotation
                image_path = annotation.image_path

                if hasattr(explorer, 'annotation_window'):
                    if explorer.annotation_window.current_image_path != image_path:
                        if hasattr(explorer.annotation_window, 'set_image'):
                            explorer.annotation_window.set_image(image_path)
                    if hasattr(explorer.annotation_window, 'select_annotation'):
                        explorer.annotation_window.select_annotation(annotation)
                    if hasattr(explorer.annotation_window, 'center_on_annotation'):
                        explorer.annotation_window.center_on_annotation(annotation)

                explorer.update_label_window_selection()
                explorer.update_button_states()
                event.accept()
                return

        # Handle left-click for selection or rubber band
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                # The viewer (controller) directly changes the state on the data item.
                is_currently_selected = item_at_pos.data_item.is_selected
                item_at_pos.data_item.set_selected(not is_currently_selected)
                item_at_pos.setSelected(not is_currently_selected)  # Keep scene selection in sync
                self.on_selection_changed()  # Manually trigger update
                return

            self.selection_at_press = set(self.graphics_scene.selectedItems())
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.rubber_band_origin = self.graphics_view.mapToScene(event.pos())
            self.rubber_band = QGraphicsRectItem(QRectF(self.rubber_band_origin, self.rubber_band_origin))
            self.rubber_band.setPen(QPen(QColor(0, 100, 255), 1, Qt.DotLine))
            self.rubber_band.setBrush(QBrush(QColor(0, 100, 255, 50)))
            self.graphics_scene.addItem(self.rubber_band)

        elif event.button() == Qt.RightButton:
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mousePressEvent(self.graphics_view, left_event)
        else:
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            QGraphicsView.mousePressEvent(self.graphics_view, event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and reset the main view."""
        if event.button() == Qt.LeftButton:
            if self.graphics_scene.selectedItems():
                self.graphics_scene.clearSelection()
            self.reset_view_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dynamic selection and panning."""
        if self.rubber_band:
            # Update the rubber band rectangle as the mouse moves
            current_pos = self.graphics_view.mapToScene(event.pos())
            self.rubber_band.setRect(QRectF(self.rubber_band_origin, current_pos).normalized())
            # Create a selection path from the rubber band rectangle
            path = QPainterPath()
            path.addRect(self.rubber_band.rect())
            # Block signals to avoid recursive selectionChanged events
            self.graphics_scene.blockSignals(True)
            self.graphics_scene.setSelectionArea(path)
            # Restore selection for items that were already selected at press
            if self.selection_at_press:
                for item in self.selection_at_press:
                    item.setSelected(True)
            self.graphics_scene.blockSignals(False)
            # Manually trigger selection changed logic
            self.on_selection_changed()
        elif event.buttons() == Qt.RightButton:
            # Forward right-drag as left-drag for panning
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mouseMoveEvent(self.graphics_view, left_event)
            self._schedule_view_update()
        else:
            # Default mouse move handling
            QGraphicsView.mouseMoveEvent(self.graphics_view, event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize the action and clean up."""
        if self.rubber_band:
            self.graphics_scene.removeItem(self.rubber_band)
            self.rubber_band = None
            self.selection_at_press = None
        elif event.button() == Qt.RightButton:
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mouseReleaseEvent(self.graphics_view, left_event)
            self._schedule_view_update()
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        else:
            QGraphicsView.mouseReleaseEvent(self.graphics_view, event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Set anchor points so zoom occurs at mouse position
        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)

        # Get the scene position before zooming
        old_pos = self.graphics_view.mapToScene(event.pos())

        # Determine zoom direction
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        # Apply zoom
        self.graphics_view.scale(zoom_factor, zoom_factor)

        # Get the scene position after zooming
        new_pos = self.graphics_view.mapToScene(event.pos())

        # Translate view to keep mouse position stable
        delta = new_pos - old_pos
        self.graphics_view.translate(delta.x(), delta.y())
        self._schedule_view_update()

    def update_embeddings(self, data_items):
        """Update the embedding visualization. Creates an EmbeddingPointItem for
        each AnnotationDataItem and links them."""
        # Reset isolation state when loading new points
        if self.isolated_mode:
            self.show_all_points()

        self.clear_points()
        for item in data_items:
            point = EmbeddingPointItem(item)
            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point
        
        # Ensure buttons are in the correct initial state
        self._update_toolbar_state()
        # Set initial visibility
        self._update_visible_points()

    def clear_points(self):
        """Clear all embedding points from the scene."""
        if self.isolated_mode:
            self.show_all_points()

        for point in self.points_by_id.values():
            self.graphics_scene.removeItem(point)
        self.points_by_id.clear()
        self._update_toolbar_state()

    def on_selection_changed(self):
        """
        Handles selection changes in the scene. Updates the central data model
        and emits a signal to notify other parts of the application.
        """
        if not self.graphics_scene:
            return
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            return

        current_selection_ids = {item.data_item.annotation.id for item in selected_items}

        if current_selection_ids != self.previous_selection_ids:
            for point_id, point in self.points_by_id.items():
                is_selected = point_id in current_selection_ids
                point.data_item.set_selected(is_selected)

            self.selection_changed.emit(list(current_selection_ids))
            self.previous_selection_ids = current_selection_ids

        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop()

        for point in self.points_by_id.values():
            if not point.isSelected():
                point.setPen(QPen(QColor("black"), POINT_WIDTH))
        if selected_items and hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.start()

        # Update button states based on new selection
        self._update_toolbar_state()
        
        # A selection change can affect visibility (e.g., deselecting an off-screen point)
        self._schedule_view_update()

    def animate_selection(self):
        """Animate selected points with a marching ants effect."""
        if not self.graphics_scene:
            return
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            return

        self.animation_offset = (self.animation_offset + 1) % 20
        for item in selected_items:
            # Get the color directly from the source of truth
            original_color = item.data_item.effective_color
            darker_color = original_color.darker(150)
            animated_pen = QPen(darker_color, POINT_WIDTH)
            animated_pen.setStyle(Qt.CustomDashLine)
            animated_pen.setDashPattern([1, 2])
            animated_pen.setDashOffset(self.animation_offset)
            item.setPen(animated_pen)

    def render_selection_from_ids(self, selected_ids):
        """
        Updates the visual selection of points based on a set of annotation IDs
        provided by an external controller.
        """
        blocker = QSignalBlocker(self.graphics_scene)

        for ann_id, point in self.points_by_id.items():
            is_selected = ann_id in selected_ids
            # 1. Update the state on the central data item
            point.data_item.set_selected(is_selected)
            # 2. Update the selection state of the graphics item itself
            point.setSelected(is_selected)

        blocker.unblock()

        # Manually trigger on_selection_changed to update animation and emit signals
        self.on_selection_changed()
        
        # After selection, update visibility to ensure newly selected points are shown
        self._update_visible_points()

    def fit_view_to_points(self):
        """Fit the view to show all embedding points."""
        if self.points_by_id:
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            self.graphics_view.fitInView(-2500, -2500, 5000, 5000, Qt.KeepAspectRatio)
            
            
class AnnotationViewer(QWidget):
    """
    Widget containing a toolbar and a scrollable grid for displaying annotation image crops.
    Implements virtualization to only render visible widgets.
    """
    selection_changed = pyqtSignal(list)
    preview_changed = pyqtSignal(list)
    reset_view_requested = pyqtSignal()
    find_similar_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the AnnotationViewer widget."""
        super(AnnotationViewer, self).__init__(parent)
        self.explorer_window = parent

        self.annotation_widgets_by_id = {}
        self.selected_widgets = []
        self.last_selected_item_id = None  # Use a persistent ID for the selection anchor
        self.current_widget_size = 96
        self.selection_at_press = set()
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        self.mouse_pressed_on_widget = False
        self.preview_label_assignments = {}
        self.original_label_assignments = {}
        self.isolated_mode = False
        self.isolated_widgets = set()

        # State for sorting options
        self.active_ordered_ids = []
        self.is_confidence_sort_available = False

        # New attributes for virtualization
        self.all_data_items = []
        self.widget_positions = {}  # ann_id -> QRect
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_visible_widgets)

        self.setup_ui()

        # Connect scrollbar value changed to schedule an update for virtualization
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._schedule_update)
        # Install an event filter on the viewport to handle mouse events for rubber band selection
        self.scroll_area.viewport().installEventFilter(self)

    def setup_ui(self):
        """Set up the UI with a toolbar and a scrollable content area."""
        # This widget is the main container with its own layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Create and add the toolbar to the main layout
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)

        self.isolate_button = QPushButton("Isolate Selection")
        self.isolate_button.setToolTip("Hide all non-selected annotations")
        self.isolate_button.clicked.connect(self.isolate_selection)
        toolbar_layout.addWidget(self.isolate_button)

        self.show_all_button = QPushButton("Show All")
        self.show_all_button.setToolTip("Show all filtered annotations")
        self.show_all_button.clicked.connect(self.show_all_annotations)
        toolbar_layout.addWidget(self.show_all_button)

        toolbar_layout.addWidget(self._create_separator())

        sort_label = QLabel("Sort By:")
        toolbar_layout.addWidget(sort_label)
        self.sort_combo = QComboBox()
        # Remove "Similarity" as it's now an implicit action
        self.sort_combo.addItems(["None", "Label", "Image", "Confidence"])
        self.sort_combo.insertSeparator(3)  # Add separator before "Confidence"
        self.sort_combo.currentTextChanged.connect(self.on_sort_changed)
        toolbar_layout.addWidget(self.sort_combo)
        
        toolbar_layout.addWidget(self._create_separator())
        
        self.find_similar_button = QToolButton()
        self.find_similar_button.setText("Find Similar")
        self.find_similar_button.setToolTip("Find annotations visually similar to the selection.")
        self.find_similar_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.find_similar_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.find_similar_button.setStyleSheet(
            "QToolButton::menu-indicator { subcontrol-position: right center; subcontrol-origin: padding; left: -4px; }"
        )

        run_similar_action = QAction("Find Similar", self)
        run_similar_action.triggered.connect(self.find_similar_requested.emit)
        self.find_similar_button.setDefaultAction(run_similar_action)
        
        self.similarity_settings_widget = SimilaritySettingsWidget()
        settings_menu = QMenu(self)
        widget_action = QWidgetAction(settings_menu)
        widget_action.setDefaultWidget(self.similarity_settings_widget)
        settings_menu.addAction(widget_action)
        self.find_similar_button.setMenu(settings_menu)
        toolbar_layout.addWidget(self.find_similar_button)
        
        toolbar_layout.addStretch()

        size_label = QLabel("Size:")
        toolbar_layout.addWidget(size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(32)
        self.size_slider.setMaximum(256)
        self.size_slider.setValue(96)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(32)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        toolbar_layout.addWidget(self.size_slider)

        self.size_value_label = QLabel("96")
        self.size_value_label.setMinimumWidth(30)
        toolbar_layout.addWidget(self.size_value_label)
        main_layout.addWidget(toolbar_widget)
        
        # Create the scroll area which will contain the content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

        # Set the initial state of the sort options
        self._update_sort_options_state()
        self._update_toolbar_state()
        
    def _create_separator(self):
        """Creates a vertical separator for the toolbar."""
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; margin: 0 5px;")
        return separator
    
    def _update_sort_options_state(self):
        """Enable/disable sort options based on available data."""
        model = self.sort_combo.model()

        # Enable/disable "Confidence" option
        confidence_item_index = self.sort_combo.findText("Confidence")
        if confidence_item_index != -1:
            model.item(confidence_item_index).setEnabled(self.is_confidence_sort_available)

    def handle_annotation_context_menu(self, widget, event):
        """Handle context menu requests (e.g., right-click) on an annotation widget."""
        if event.modifiers() == Qt.ControlModifier:
            explorer = self.explorer_window
            image_path = widget.annotation.image_path
            annotation_to_select = widget.annotation
        
            # ctrl+right click to only select this annotation (single selection):
            self.clear_selection()  
            self.select_widget(widget)
            changed_ids = [widget.data_item.annotation.id]

            if changed_ids:
                self.selection_changed.emit(changed_ids)

            if hasattr(explorer, 'annotation_window'):
                # Check if the image needs to be changed
                if explorer.annotation_window.current_image_path != image_path:
                    if hasattr(explorer.annotation_window, 'set_image'):
                        explorer.annotation_window.set_image(image_path)

                # Now, select the annotation in the annotation_window (activates animation)
                if hasattr(explorer.annotation_window, 'select_annotation'):
                    explorer.annotation_window.select_annotation(annotation_to_select, quiet_mode=True)
                    
                # Center the annotation window view on the selected annotation
                if hasattr(explorer.annotation_window, 'center_on_annotation'):
                    explorer.annotation_window.center_on_annotation(annotation_to_select)
                    
                # Show resize handles for Rectangle annotations
                if isinstance(annotation_to_select, RectangleAnnotation):
                    explorer.annotation_window.set_selected_tool('select')  # Accidentally unselects in AnnotationWindow
                    explorer.annotation_window.select_annotation(annotation_to_select, quiet_mode=True)
                    select_tool = explorer.annotation_window.tools.get('select')

                    if select_tool:
                        # Engage the selection lock.
                        select_tool.selection_locked = True
                        # Show the resize handles for the now-selected annotation.
                        select_tool._show_resize_handles()

                # Also clear any existing selection in the explorer window itself
                explorer.embedding_viewer.render_selection_from_ids({widget.data_item.annotation.id})
                explorer.update_label_window_selection()
                explorer.update_button_states()
            
            event.accept()

    @pyqtSlot()
    def isolate_selection(self):
        """Hides all annotation widgets that are not currently selected."""
        if not self.selected_widgets:
            return

        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                if widget not in self.isolated_widgets:
                    widget.hide()
            self.isolated_mode = True
            self.recalculate_layout()
        finally:
            self.content_widget.setUpdatesEnabled(True)

        self._update_toolbar_state()
        self.explorer_window.main_window.label_window.update_annotation_count()

    def isolate_and_select_from_ids(self, ids_to_isolate):
        """
        Enters isolated mode showing only widgets for the given IDs, and also
        selects them. This is the primary entry point from external viewers.
        The isolated set is 'sticky' and will not change on subsequent internal
        selection changes.
        """
        # Get the widget objects from the IDs
        widgets_to_isolate = {
            self.annotation_widgets_by_id[ann_id]
            for ann_id in ids_to_isolate
            if ann_id in self.annotation_widgets_by_id
        }

        if not widgets_to_isolate:
            return

        self.isolated_widgets = widgets_to_isolate
        self.isolated_mode = True

        self.render_selection_from_ids(ids_to_isolate)
        self.recalculate_layout()

    def display_and_isolate_ordered_results(self, ordered_ids):
        """
        Isolates the view to a specific set of ordered widgets, ensuring the 
        grid is always updated. This is the new primary method for showing 
        similarity results.
        """
        self.active_ordered_ids = ordered_ids
        
        # Render the selection based on the new order
        self.render_selection_from_ids(set(ordered_ids)) 

        # Now, perform the isolation logic directly to bypass the guard clause
        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                # Show widget if it's in our target set, hide otherwise
                if widget in self.isolated_widgets:
                    widget.show()
                else:
                    widget.hide()
                    
            self.isolated_mode = True
            self.recalculate_layout()  # Crucial grid update
        finally:
            self.content_widget.setUpdatesEnabled(True)

        self._update_toolbar_state()
        self.explorer_window.main_window.label_window.update_annotation_count()

    @pyqtSlot()
    def show_all_annotations(self):
        """Shows all annotation widgets, exiting the isolated mode."""
        if not self.isolated_mode:
            return

        self.isolated_mode = False
        self.isolated_widgets.clear()
        self.active_ordered_ids = []  # Clear similarity sort context

        self.content_widget.setUpdatesEnabled(False)
        try:
            # Show all widgets that are managed by the viewer
            for widget in self.annotation_widgets_by_id.values():
                widget.show()

            self.recalculate_layout()
        finally:
            self.content_widget.setUpdatesEnabled(True)

        self._update_toolbar_state()
        self.explorer_window.main_window.label_window.update_annotation_count()

    def _update_toolbar_state(self):
        """Updates the toolbar buttons based on selection and isolation mode."""
        selection_exists = bool(self.selected_widgets)
        if self.isolated_mode:
            self.isolate_button.hide()
            self.show_all_button.show()
            self.show_all_button.setEnabled(True)
        else:
            self.isolate_button.show()
            self.show_all_button.hide()
            self.isolate_button.setEnabled(selection_exists)

    def on_sort_changed(self, sort_type):
        """Handle sort type change."""
        self.active_ordered_ids = []  # Clear any special ordering
        self.recalculate_layout()

    def set_confidence_sort_availability(self, is_available):
        """Sets the availability of the confidence sort option."""
        self.is_confidence_sort_available = is_available
        self._update_sort_options_state()

    def _get_sorted_data_items(self):
        """Get data items sorted according to the current sort setting."""
        # If a specific order is active (e.g., from similarity search), use it.
        if self.active_ordered_ids:
            item_map = {i.annotation.id: i for i in self.all_data_items}
            ordered_items = [item_map[ann_id] for ann_id in self.active_ordered_ids if ann_id in item_map]
            return ordered_items

        # Otherwise, use the dropdown sort logic
        sort_type = self.sort_combo.currentText()
        items = list(self.all_data_items)

        if sort_type == "Label":
            items.sort(key=lambda i: i.effective_label.short_label_code)
        elif sort_type == "Image":
            items.sort(key=lambda i: os.path.basename(i.annotation.image_path))
        elif sort_type == "Confidence":
            # Sort by confidence, descending. Handles cases with no confidence gracefully.
            items.sort(key=lambda i: i.get_effective_confidence(), reverse=True)

        return items

    def _get_sorted_widgets(self):
        """
        Get widgets sorted according to the current sort setting.
        This is kept for compatibility with selection logic.
        """
        sorted_data_items = self._get_sorted_data_items()
        return [self.annotation_widgets_by_id[item.annotation.id]
                for item in sorted_data_items if item.annotation.id in self.annotation_widgets_by_id]

    def _group_data_items_by_sort_key(self, data_items):
        """Group data items by the current sort key."""
        sort_type = self.sort_combo.currentText()
        if not self.active_ordered_ids and sort_type == "None":
            return [("", data_items)]

        if self.active_ordered_ids:  # Don't show group headers for similarity results
            return [("", data_items)]

        groups = []
        current_group = []
        current_key = None
        for item in data_items:
            if sort_type == "Label":
                key = item.effective_label.short_label_code
            elif sort_type == "Image":
                key = os.path.basename(item.annotation.image_path)
            else:
                key = "" # No headers for Confidence or None
            
            if key and current_key != key:
                if current_group:
                    groups.append((current_key, current_group))
                current_group = [item]
                current_key = key
            else:
                current_group.append(item)
        if current_group:
            groups.append((current_key, current_group))
        return groups

    def _clear_separator_labels(self):
        """Remove any existing group header labels."""
        if hasattr(self, '_group_headers'):
            for header in self._group_headers:
                header.setParent(None)
                header.deleteLater()
        self._group_headers = []

    def _create_group_header(self, text):
        """Create a group header label."""
        if not hasattr(self, '_group_headers'):
            self._group_headers = []
        header = QLabel(text, self.content_widget)
        header.setStyleSheet(
            "QLabel {"
            " font-weight: bold;"
            " font-size: 12px;"
            " color: #555;"
            " background-color: #f0f0f0;"
            " border: 1px solid #ccc;"
            " border-radius: 3px;"
            " padding: 5px 8px;"
            " margin: 2px 0px;"
            " }"
        )
        header.setFixedHeight(30)
        header.setMinimumWidth(self.scroll_area.viewport().width() - 20)
        header.show()
        self._group_headers.append(header)
        return header

    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
        if value % 2 != 0:
            value -= 1

        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        self.recalculate_layout()

    def _schedule_update(self):
        """Schedules a delayed update of visible widgets to avoid performance issues during rapid scrolling."""
        self.update_timer.start(50)  # 50ms delay

    def _update_visible_widgets(self):
        """Shows and loads widgets that are in the viewport, and hides/unloads others."""
        if not self.widget_positions:
            return

        self.content_widget.setUpdatesEnabled(False)
        
        # Determine the visible rectangle in the content widget's coordinates
        scroll_y = self.scroll_area.verticalScrollBar().value()
        visible_content_rect = QRect(0, 
                                     scroll_y, 
                                     self.scroll_area.viewport().width(), 
                                     self.scroll_area.viewport().height())

        # Add a buffer to load images slightly before they become visible
        buffer = self.scroll_area.viewport().height() // 2
        visible_content_rect.adjust(0, -buffer, 0, buffer)

        visible_ids = set()
        for ann_id, rect in self.widget_positions.items():
            if rect.intersects(visible_content_rect):
                visible_ids.add(ann_id)

        # Update widgets based on visibility
        for ann_id, widget in self.annotation_widgets_by_id.items():
            if ann_id in visible_ids:
                # This widget should be visible
                widget.setGeometry(self.widget_positions[ann_id])
                widget.load_image()  # Lazy-loads the image
                widget.show()
            else:
                # This widget is not visible
                if widget.isVisible():
                    widget.hide()
                    widget.unload_image()  # Free up memory

        self.content_widget.setUpdatesEnabled(True)

    def recalculate_layout(self):
        """Calculates the positions for all widgets and the total size of the content area."""
        if not self.all_data_items:
            self.content_widget.setMinimumSize(1, 1)
            return

        self._clear_separator_labels()
        sorted_data_items = self._get_sorted_data_items()

        # If in isolated mode, only consider the isolated widgets for layout
        if self.isolated_mode:
            isolated_ids = {w.data_item.annotation.id for w in self.isolated_widgets}
            sorted_data_items = [item for item in sorted_data_items if item.annotation.id in isolated_ids]

        if not sorted_data_items:
            self.content_widget.setMinimumSize(1, 1)
            return

        # Create groups based on the current sort key
        groups = self._group_data_items_by_sort_key(sorted_data_items)
        spacing = max(5, int(self.current_widget_size * 0.08))
        available_width = self.scroll_area.viewport().width()
        x, y = spacing, spacing
        max_height_in_row = 0

        self.widget_positions.clear()

        # Calculate positions
        for group_name, group_data_items in groups:
            if group_name and self.sort_combo.currentText() != "None":
                if x > spacing:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                header_label = self._create_group_header(group_name)
                header_label.move(x, y)
                y += header_label.height() + spacing
                x = spacing
                max_height_in_row = 0

            for data_item in group_data_items:
                ann_id = data_item.annotation.id
                if ann_id in self.annotation_widgets_by_id:
                    widget = self.annotation_widgets_by_id[ann_id]
                    # Make sure this is present:
                    widget.update_height(self.current_widget_size)
                else:
                    widget = AnnotationImageWidget(data_item, self.current_widget_size, self, self.content_widget)
                    # Ensure aspect ratio is calculated on creation:
                    widget.recalculate_aspect_ratio() 
                    self.annotation_widgets_by_id[ann_id] = widget

                widget_size = widget.size()
                if x > spacing and x + widget_size.width() > available_width:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0

                self.widget_positions[ann_id] = QRect(x, y, widget_size.width(), widget_size.height())

                x += widget_size.width() + spacing
                max_height_in_row = max(max_height_in_row, widget_size.height())

        total_height = y + max_height_in_row + spacing
        self.content_widget.setMinimumSize(available_width, total_height)

        # After calculating layout, update what's visible
        self._update_visible_widgets()

    def update_annotations(self, data_items):
        """Update displayed annotations, creating new widgets for them."""
        if self.isolated_mode:
            self.show_all_annotations()

        # Clear out widgets for data items that are no longer in the new set
        all_ann_ids = {item.annotation.id for item in data_items}
        for ann_id, widget in list(self.annotation_widgets_by_id.items()):
            if ann_id not in all_ann_ids:
                if widget in self.selected_widgets:
                    self.selected_widgets.remove(widget)
                widget.setParent(None)
                widget.deleteLater()
                del self.annotation_widgets_by_id[ann_id]

        self.all_data_items = data_items
        self.selected_widgets.clear()
        self.last_selected_item_id = None

        self.recalculate_layout()
        self._update_toolbar_state()
        # Update the label window with the new annotation count
        self.explorer_window.main_window.label_window.update_annotation_count()

    def resizeEvent(self, event):
        """On window resize, reflow the annotation widgets."""
        super(AnnotationViewer, self).resizeEvent(event)
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self.recalculate_layout)
        self._resize_timer.start(100)

    def keyPressEvent(self, event):
        """Handles key presses for deleting selected annotations."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and event.modifiers() == Qt.ControlModifier:
            if not self.selected_widgets:
                super().keyPressEvent(event)
                return

            # Extract the central data items from the selected widgets
            data_items_to_delete = [widget.data_item for widget in self.selected_widgets]

            # Delegate the actual deletion to the main ExplorerWindow
            if data_items_to_delete:
                self.explorer_window.delete_data_items(data_items_to_delete)

            event.accept()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, source, event):
        """Filters events from the scroll area's viewport to handle mouse interactions."""
        if source is self.scroll_area.viewport():
            if event.type() == QEvent.MouseButtonPress:
                return self.viewport_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self.viewport_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self.viewport_mouse_release(event)
            elif event.type() == QEvent.MouseButtonDblClick:
                return self.viewport_mouse_double_click(event)

        return super(AnnotationViewer, self).eventFilter(source, event)

    def viewport_mouse_press(self, event):
        """Handle mouse press inside the viewport for selection."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # Start rubber band selection
            self.selection_at_press = set(self.selected_widgets)
            self.rubber_band_origin = event.pos()

            # Check if the press was on a widget to avoid starting rubber band on a widget click
            content_pos = self.content_widget.mapFrom(self.scroll_area.viewport(), event.pos())
            child_at_pos = self.content_widget.childAt(content_pos)
            self.mouse_pressed_on_widget = isinstance(child_at_pos, AnnotationImageWidget)

            return True  # Event handled

        elif event.button() == Qt.LeftButton and not event.modifiers():
            # Clear selection if clicking on the background
            content_pos = self.content_widget.mapFrom(self.scroll_area.viewport(), event.pos())
            if self.content_widget.childAt(content_pos) is None:
                if self.selected_widgets:
                    changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                    self.clear_selection()
                    self.selection_changed.emit(changed_ids)
                if hasattr(self.explorer_window.annotation_window, 'unselect_annotations'):
                    self.explorer_window.annotation_window.unselect_annotations()
                return True

        return False  # Let the event propagate for default behaviors like scrolling

    def viewport_mouse_double_click(self, event):
        """Handle double-click in the viewport to clear selection and reset view."""
        if event.button() == Qt.LeftButton:
            if self.selected_widgets:
                changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                self.clear_selection()
                self.selection_changed.emit(changed_ids)
            if self.isolated_mode:
                self.show_all_annotations()
            self.reset_view_requested.emit()
            return True
        return False

    def viewport_mouse_move(self, event):
        """Handle mouse move in the viewport for dynamic rubber band selection."""
        if (
            self.rubber_band_origin is None or
            event.buttons() != Qt.LeftButton or
            event.modifiers() != Qt.ControlModifier or
            self.mouse_pressed_on_widget
        ):
            return False

        # Only start selection if drag distance exceeds threshold
        distance = (event.pos() - self.rubber_band_origin).manhattanLength()
        if distance < self.drag_threshold:
            return True

        # Create and show the rubber band if not already present
        if not self.rubber_band:
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.scroll_area.viewport())

        rect = QRect(self.rubber_band_origin, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()

        selection_rect = self.rubber_band.geometry()
        content_widget = self.content_widget
        changed_ids = []

        # Iterate over all annotation widgets to update selection state
        for widget in self.annotation_widgets_by_id.values():
            # Map widget's geometry from content_widget coordinates to viewport coordinates
            mapped_top_left = content_widget.mapTo(self.scroll_area.viewport(), widget.geometry().topLeft())
            widget_rect_in_viewport = QRect(mapped_top_left, widget.geometry().size())
            
            is_in_band = selection_rect.intersects(widget_rect_in_viewport)
            should_be_selected = (widget in self.selection_at_press) or is_in_band

            # Select or deselect widgets as needed
            if should_be_selected and not widget.is_selected():
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)

            elif not should_be_selected and widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)

        # Emit signal if any selection state changed
        if changed_ids:
            self.selection_changed.emit(changed_ids)

        return True

    def viewport_mouse_release(self, event):
        """Handle mouse release in the viewport to finalize rubber band selection."""
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None
            self.rubber_band_origin = None
            return True
        return False

    def handle_annotation_selection(self, widget, event):
        """Handle selection of annotation widgets with different modes (single, ctrl, shift)."""
        # The list for range selection should be based on the sorted data items
        sorted_data_items = self._get_sorted_data_items()

        # In isolated mode, the list should only contain isolated items
        if self.isolated_mode:
            isolated_ids = {w.data_item.annotation.id for w in self.isolated_widgets}
            sorted_data_items = [item for item in sorted_data_items if item.annotation.id in isolated_ids]

        try:
            # Find the index of the clicked widget's data item
            widget_data_item = widget.data_item
            current_index = sorted_data_items.index(widget_data_item)
        except ValueError:
            return

        modifiers = event.modifiers()
        changed_ids = []

        # Shift or Shift+Ctrl: range selection.
        if modifiers in (Qt.ShiftModifier, Qt.ShiftModifier | Qt.ControlModifier):
            last_index = -1
            if self.last_selected_item_id:
                try:
                    # Find the data item corresponding to the last selected ID
                    last_item = self.explorer_window.data_item_cache[self.last_selected_item_id]
                    # Find its index in the *current* sorted list
                    last_index = sorted_data_items.index(last_item)
                except (KeyError, ValueError):
                    # The last selected item is not in the current view or cache, so no anchor
                    last_index = -1

            if last_index != -1:
                start = min(last_index, current_index)
                end = max(last_index, current_index)

                # Select all widgets in the range
                for i in range(start, end + 1):
                    item_to_select = sorted_data_items[i]
                    widget_to_select = self.annotation_widgets_by_id.get(item_to_select.annotation.id)
                    if widget_to_select and self.select_widget(widget_to_select):
                        changed_ids.append(item_to_select.annotation.id)
            else:
                # No previous selection, just select the clicked widget
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)

            self.last_selected_item_id = widget.data_item.annotation.id

        # Ctrl: toggle selection of the clicked widget
        elif modifiers == Qt.ControlModifier:
            # Toggle selection and update the anchor
            if self.toggle_widget_selection(widget):
                changed_ids.append(widget.data_item.annotation.id)
            self.last_selected_item_id = widget.data_item.annotation.id

        # No modifier: single selection
        else:
            newly_selected_id = widget.data_item.annotation.id

            # Deselect all others
            for w in list(self.selected_widgets):
                if w.data_item.annotation.id != newly_selected_id:
                    if self.deselect_widget(w):
                        changed_ids.append(w.data_item.annotation.id)

            # Select the clicked widget
            if self.select_widget(widget):
                changed_ids.append(newly_selected_id)
            self.last_selected_item_id = widget.data_item.annotation.id

        # If in isolated mode, update which widgets are visible
        if self.isolated_mode:
            pass  # Do not change the isolated set on internal selection changes

        # Emit signal if any selection state changed
        if changed_ids:
            self.selection_changed.emit(changed_ids)

    def toggle_widget_selection(self, widget):
        """Toggles the selection state of a widget and returns True if changed."""
        if widget.is_selected():
            return self.deselect_widget(widget)
        else:
            return self.select_widget(widget)

    def select_widget(self, widget):
        """Selects a widget, updates its data_item, and returns True if state changed."""
        if not widget.is_selected():  # is_selected() checks the data_item
            # 1. Controller modifies the state on the data item
            widget.data_item.set_selected(True)
            # 2. Controller tells the view to update its appearance
            widget.update_selection_visuals()
            self.selected_widgets.append(widget)
            self._update_toolbar_state()
            return True
        return False

    def deselect_widget(self, widget):
        """Deselects a widget, updates its data_item, and returns True if state changed."""
        if widget.is_selected():
            # 1. Controller modifies the state on the data item
            widget.data_item.set_selected(False)
            # 2. Controller tells the view to update its appearance
            widget.update_selection_visuals()
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            self._update_toolbar_state()
            return True
        return False

    def clear_selection(self):
        """Clear all selected widgets and update toolbar state."""
        for widget in list(self.selected_widgets):
            # This will internally call deselect_widget, which is fine
            self.deselect_widget(widget)

        self.selected_widgets.clear()
        self._update_toolbar_state()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]

    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of widgets based on a set of IDs from the controller."""
        self.setUpdatesEnabled(False)
        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in selected_ids
                # 1. Update the state on the central data item
                widget.data_item.set_selected(is_selected)
                # 2. Tell the widget to update its visuals based on the new state
                widget.update_selection_visuals()

            # Resync internal list of selected widgets from the source of truth
            self.selected_widgets = [w for w in self.annotation_widgets_by_id.values() if w.is_selected()]

        finally:
            self.setUpdatesEnabled(True)
        self._update_toolbar_state()

    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label and emit a signal for the embedding view to update."""
        if not self.selected_widgets or not preview_label:
            return
        changed_ids = []
        for widget in self.selected_widgets:
            widget.data_item.set_preview_label(preview_label)
            widget.update()  # Force repaint with new color
            changed_ids.append(widget.data_item.annotation.id)

        if self.sort_combo.currentText() == "Label":
            self.recalculate_layout()
        if changed_ids:
            self.preview_changed.emit(changed_ids)

    def clear_preview_states(self):
        """
        Clears all preview states, including label changes,
        reverting them to their original state.
        """
        something_changed = False
        for widget in self.annotation_widgets_by_id.values():
            # Check for and clear preview labels
            if widget.data_item.has_preview_changes():
                widget.data_item.clear_preview_label()
                widget.update()  # Repaint to show original color
                something_changed = True

        if something_changed:
            # Recalculate positions to update sorting and re-flow the layout
            if self.sort_combo.currentText() == "Label":
                self.recalculate_layout()

    def has_preview_changes(self):
        """Return True if there are preview changes."""
        return any(w.data_item.has_preview_changes() for w in self.annotation_widgets_by_id.values())

    def get_preview_changes_summary(self):
        """Get a summary of preview changes."""
        change_count = sum(1 for w in self.annotation_widgets_by_id.values() if w.data_item.has_preview_changes())
        return f"{change_count} annotation(s) with preview changes" if change_count else "No preview changes"

    def apply_preview_changes_permanently(self):
        """Apply preview changes permanently."""
        applied_annotations = []
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.apply_preview_permanently():
                applied_annotations.append(widget.annotation)
        return applied_annotations
