class EmbeddingViewer(QWidget):  # Change inheritance to QWidget
    """Custom QGraphicsView for interactive embedding visualization with zooming, panning, and selection."""
    
    # Define signal to report selection changes
    selection_changed = pyqtSignal(list)  # list of all currently selected annotation IDs
    reset_view_requested = pyqtSignal()  # Signal to reset the view to fit all points
    
    def __init__(self, parent=None):
        # Create the graphics scene first
        self.graphics_scene = QGraphicsScene()
        self.graphics_scene.setSceneRect(-5000, -5000, 10000, 10000)
        
        # Initialize as a QWidget
        super(EmbeddingViewer, self).__init__(parent)
        self.explorer_window = parent
        
        # Create the actual graphics view
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setMinimumHeight(200)
        
        # Custom rubber_band state variables
        self.rubber_band = None
        self.rubber_band_origin = QPointF()
        self.selection_at_press = None
        
        self.points_by_id = {}  # Map annotation ID to embedding point
        self.previous_selection_ids = set()  # Track previous selection to detect changes
    
        self.animation_offset = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_selection)
        self.animation_timer.setInterval(100)
        
        # Connect the scene's selection signal
        self.graphics_scene.selectionChanged.connect(self.on_selection_changed)
        
        # Setup the UI with header
        self.setup_ui()
        
        # Connect mouse events to the graphics view
        self.graphics_view.mousePressEvent = self.mousePressEvent
        self.graphics_view.mouseDoubleClickEvent = self.mouseDoubleClickEvent
        self.graphics_view.mouseReleaseEvent = self.mouseReleaseEvent
        self.graphics_view.mouseMoveEvent = self.mouseMoveEvent
        self.graphics_view.wheelEvent = self.wheelEvent

    def setup_ui(self):
        """Set up the UI with header layout and graphics view."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header layout
        header_layout = QHBoxLayout()
        
        # Home button
        self.home_button = QPushButton("Home")
        self.home_button.setToolTip("Reset view to fit all points")
        self.home_button.clicked.connect(self.reset_view)
        header_layout.addWidget(self.home_button)
        
        # Add stretch to push future controls to the right if needed
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Add the graphics view
        layout.addWidget(self.graphics_view)
        # Add a placeholder label when no embedding is available
        self.placeholder_label = QLabel(
            "No embedding data available.\nPress 'Apply Embedding' to generate visualization."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("color: gray; font-size: 14px;")
        layout.addWidget(self.placeholder_label)
        
        # Initially show placeholder
        self.show_placeholder()

    def reset_view(self):
        """Reset the view to fit all embedding points."""
        self.fit_view_to_points()

    def show_placeholder(self):
        """Show the placeholder message and hide the graphics view."""
        self.graphics_view.setVisible(False)
        self.placeholder_label.setVisible(True)
        self.home_button.setEnabled(False)

    def show_embedding(self):
        """Show the graphics view and hide the placeholder message."""
        self.graphics_view.setVisible(True)
        self.placeholder_label.setVisible(False)
        self.home_button.setEnabled(True)

    # Delegate graphics view methods
    def setRenderHint(self, hint):
        self.graphics_view.setRenderHint(hint)
    
    def setDragMode(self, mode):
        self.graphics_view.setDragMode(mode)
    
    def setTransformationAnchor(self, anchor):
        self.graphics_view.setTransformationAnchor(anchor)
    
    def setResizeAnchor(self, anchor):
        self.graphics_view.setResizeAnchor(anchor)
    
    def mapToScene(self, point):
        return self.graphics_view.mapToScene(point)
    
    def scale(self, sx, sy):
        self.graphics_view.scale(sx, sy)
    
    def translate(self, dx, dy):
        self.graphics_view.translate(dx, dy)
    
    def fitInView(self, rect, aspect_ratio):
        self.graphics_view.fitInView(rect, aspect_ratio)

    def mousePressEvent(self, event):
        """Handle mouse press for selection (point or rubber band) and panning."""
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # Check if the click is on an existing point
            item_at_pos = self.graphics_view.itemAt(event.pos())
            if isinstance(item_at_pos, EmbeddingPointItem):
                # If so, toggle its selection state and do nothing else
                self.graphics_view.setDragMode(QGraphicsView.NoDrag)
                item_at_pos.setSelected(not item_at_pos.isSelected())
                return  # Event handled

            # If the click was on the background, proceed with rubber band selection
            self.selection_at_press = set(self.graphics_scene.selectedItems())
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            self.rubber_band_origin = self.graphics_view.mapToScene(event.pos())
            self.rubber_band = QGraphicsRectItem(QRectF(self.rubber_band_origin, self.rubber_band_origin))
            self.rubber_band.setPen(QPen(QColor(0, 100, 255), 1, Qt.DotLine))
            self.rubber_band.setBrush(QBrush(QColor(0, 100, 255, 50)))
            self.graphics_scene.addItem(self.rubber_band)

        elif event.button() == Qt.RightButton:
            # Handle panning
            self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
            left_event = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mousePressEvent(self.graphics_view, left_event)
        else:
            # Handle standard single-item selection
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
            QGraphicsView.mousePressEvent(self.graphics_view, event)
            
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and reset the main view."""
        if event.button() == Qt.LeftButton:
            # Clear selection if any items are selected
            if self.graphics_scene.selectedItems():
                self.graphics_scene.clearSelection()  # This triggers on_selection_changed
            
            # Signal the main window to revert from isolation mode
            self.reset_view_requested.emit()
            event.accept()
        else:
            # Pass other double-clicks to the base class
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for dynamic selection and panning."""
        if self.rubber_band:
            # Update the rubber band geometry
            current_pos = self.graphics_view.mapToScene(event.pos())
            self.rubber_band.setRect(QRectF(self.rubber_band_origin, current_pos).normalized())
            
            path = QPainterPath()
            path.addRect(self.rubber_band.rect())

            # Block signals to perform a compound selection operation
            self.graphics_scene.blockSignals(True)

            # 1. Perform the "fancy" dynamic selection, which replaces the current selection
            #    with only the items inside the rubber band.
            self.graphics_scene.setSelectionArea(path)
            
            # 2. Add back the items that were selected at the start of the drag.
            if self.selection_at_press:
                for item in self.selection_at_press:
                    item.setSelected(True)
            
            # Unblock signals and manually trigger our handler to process the final result.
            self.graphics_scene.blockSignals(False)
            self.on_selection_changed()

        elif event.buttons() == Qt.RightButton:
            # Handle right-click panning
            left_event = QMouseEvent(event.type(), 
                                     event.localPos(), 
                                     Qt.LeftButton, 
                                     Qt.LeftButton, 
                                     event.modifiers())
            QGraphicsView.mouseMoveEvent(self.graphics_view, left_event)
        else:
            QGraphicsView.mouseMoveEvent(self.graphics_view, event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize the action and clean up."""
        if self.rubber_band:
            # Clean up the visual rectangle
            self.graphics_scene.removeItem(self.rubber_band)
            self.rubber_band = None

            # Clean up the stored selection state.
            self.selection_at_press = None
            
        elif event.button() == Qt.RightButton:
            # Finalize the pan
            left_event = QMouseEvent(event.type(), 
                                     event.localPos(), 
                                     Qt.LeftButton, 
                                     Qt.LeftButton, 
                                     event.modifiers())
            QGraphicsView.mouseReleaseEvent(self.graphics_view, left_event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        else:
            # Finalize a single click
            QGraphicsView.mouseReleaseEvent(self.graphics_view, event)
            self.graphics_view.setDragMode(QGraphicsView.NoDrag)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        self.graphics_view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.graphics_view.setResizeAnchor(QGraphicsView.NoAnchor)

        old_pos = self.graphics_view.mapToScene(event.pos())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.graphics_view.scale(zoom_factor, zoom_factor)
        new_pos = self.graphics_view.mapToScene(event.pos())
        
        delta = new_pos - old_pos
        self.graphics_view.translate(delta.x(), delta.y())

    def update_embeddings(self, data_items):
        """Update the embedding visualization with new data.

        Args:
            data_items: List of AnnotationDataItem objects.
        """
        self.clear_points()
        
        for item in data_items:
            point = EmbeddingPointItem(0, 0, POINT_SIZE, POINT_SIZE)
            point.setPos(item.embedding_x, item.embedding_y)
            
            # No need to set initial brush - paint() will handle it
            point.setPen(QPen(QColor("black"), POINT_WIDTH))
            
            point.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            point.setFlag(QGraphicsItem.ItemIsSelectable)
            
            # This is the crucial link: store the shared AnnotationDataItem
            point.setData(0, item)
            
            self.graphics_scene.addItem(point)
            self.points_by_id[item.annotation.id] = point
            
    def clear_points(self):
        """Clear all embedding points from the scene."""
        for point in self.points_by_id.values():
            self.graphics_scene.removeItem(point)
        self.points_by_id.clear()

    def on_selection_changed(self):
        """Handle point selection changes and emit a signal to the controller."""
        # Check if graphics_scene still exists and is valid
        if not self.graphics_scene or not hasattr(self.graphics_scene, 'selectedItems'):
            return
            
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            # Scene has been deleted
            return
            
        current_selection_ids = {item.data(0).annotation.id for item in selected_items}

        # If the selection has actually changed, update the model and emit
        if current_selection_ids != self.previous_selection_ids:
            # Update the central model (the AnnotationDataItem) for all points
            for point_id, point in self.points_by_id.items():
                is_selected = point_id in current_selection_ids
                point.data(0).set_selected(is_selected)

            # Emit the complete list of currently selected IDs
            self.selection_changed.emit(list(current_selection_ids))
            self.previous_selection_ids = current_selection_ids

        # Handle local animation - check if animation_timer still exists
        if hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.stop()
            
        for point in self.points_by_id.values():
            if not point.isSelected():
                point.setPen(QPen(QColor("black"), POINT_WIDTH))
        
        if selected_items and hasattr(self, 'animation_timer') and self.animation_timer:
            self.animation_timer.start()

    def animate_selection(self):
        """Animate selected points with marching ants effect using darker versions of point colors."""
        # Check if graphics_scene still exists and is valid
        if not self.graphics_scene or not hasattr(self.graphics_scene, 'selectedItems'):
            return
            
        try:
            selected_items = self.graphics_scene.selectedItems()
        except RuntimeError:
            # Scene has been deleted
            return
            
        self.animation_offset = (self.animation_offset + 1) % 20
        
        # This logic remains the same. It applies the custom pen to the selected items.
        # Because the items are EmbeddingPointItem, the default selection box won't be drawn.
        for item in selected_items:
            original_color = item.brush().color()
            darker_color = original_color.darker(150)

            animated_pen = QPen(darker_color, POINT_WIDTH)
            animated_pen.setStyle(Qt.CustomDashLine)
            animated_pen.setDashPattern([1, 2])
            animated_pen.setDashOffset(self.animation_offset)
            
            item.setPen(animated_pen)
            
    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of points based on a set of IDs from the controller."""
        # Block this scene's own selectionChanged signal to prevent an infinite loop
        blocker = QSignalBlocker(self.graphics_scene)
        
        for ann_id, point in self.points_by_id.items():
            point.setSelected(ann_id in selected_ids)
            
        self.previous_selection_ids = selected_ids
        
        # Trigger animation update
        self.on_selection_changed()

    def fit_view_to_points(self):
        """Fit the view to show all embedding points."""
        if self.points_by_id:
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            # If no points, reset to default view
            self.graphics_view.fitInView(-2500, -2500, 5000, 5000, Qt.KeepAspectRatio)
            

class AnnotationViewer(QScrollArea):
    """Scrollable grid widget for displaying annotation image crops with selection,
    filtering, and isolation support.
    """
    
    # Define signals to report changes to the ExplorerWindow
    selection_changed = pyqtSignal(list)  # list of changed annotation IDs
    preview_changed = pyqtSignal(list)   # list of annotation IDs with new previews
    reset_view_requested = pyqtSignal()  # Signal to reset the view to fit all points

    def __init__(self, parent=None):
        super(AnnotationViewer, self).__init__(parent)
        self.annotation_widgets_by_id = {}
        self.selected_widgets = []
        self.last_selected_index = -1
        self.current_widget_size = 96
        
        self.selection_at_press = set()
        self.rubber_band = None
        self.rubber_band_origin = None
        self.drag_threshold = 5
        self.mouse_pressed_on_widget = False
        
        self.preview_label_assignments = {}
        self.original_label_assignments = {}
        
        # New state variables for Isolate/Focus mode
        self.isolated_mode = False
        self.isolated_widgets = set()

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI with a toolbar and a scrollable content area."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Main container and layout
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)  # Add a little space between toolbar and content

        # --- New Toolbar ---
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)

        # Isolate/Focus controls
        self.isolate_button = QPushButton("Isolate Selection")
        isolate_icon = get_icon("focus.png")
        if not isolate_icon.isNull():
            self.isolate_button.setIcon(isolate_icon)
        self.isolate_button.setToolTip("Hide all non-selected annotations")
        self.isolate_button.clicked.connect(self.isolate_selection)
        toolbar_layout.addWidget(self.isolate_button)

        self.show_all_button = QPushButton("Show All")
        show_all_icon = get_icon("show_all.png")
        if not show_all_icon.isNull():
            self.show_all_button.setIcon(show_all_icon)
        self.show_all_button.setToolTip("Show all filtered annotations")
        self.show_all_button.clicked.connect(self.show_all_annotations)
        toolbar_layout.addWidget(self.show_all_button)

        # Add a separator
        toolbar_layout.addWidget(self._create_separator())

        # Sort controls
        sort_label = QLabel("Sort By:")
        toolbar_layout.addWidget(sort_label)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["None", "Label", "Image"])
        self.sort_combo.currentTextChanged.connect(self.on_sort_changed)
        toolbar_layout.addWidget(self.sort_combo)

        # Add a spacer to push the size controls to the right
        toolbar_layout.addStretch()

        # Size controls
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
        
        # --- Content Area ---
        self.content_widget = QWidget()
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content_scroll.setWidget(self.content_widget)
        
        main_layout.addWidget(content_scroll)
        self.setWidget(main_container)

        # Set the initial state of the toolbar buttons
        self._update_toolbar_state()
            
    @pyqtSlot()
    def isolate_selection(self):
        """Hides all annotation widgets that are not currently selected."""
        if not self.selected_widgets or self.isolated_mode:
            return

        self.isolated_widgets = set(self.selected_widgets)
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                if widget not in self.isolated_widgets:
                    widget.hide()
            self.isolated_mode = True
            self.recalculate_widget_positions()
        finally:
            self.content_widget.setUpdatesEnabled(True)

        self._update_toolbar_state()

    @pyqtSlot()
    def show_all_annotations(self):
        """Shows all annotation widgets, exiting the isolated mode."""
        if not self.isolated_mode:
            return
            
        self.isolated_mode = False
        self.isolated_widgets.clear()
        
        self.content_widget.setUpdatesEnabled(False)
        try:
            for widget in self.annotation_widgets_by_id.values():
                widget.show()
            self.recalculate_widget_positions()
        finally:
            self.content_widget.setUpdatesEnabled(True)
            
        self._update_toolbar_state()
        
    def _update_toolbar_state(self):
        """Updates the visibility and enabled state of the toolbar buttons
        based on the current selection and isolation mode.
        """
        selection_exists = bool(self.selected_widgets)
        
        if self.isolated_mode:
            self.isolate_button.hide()
            self.show_all_button.show()
            self.show_all_button.setEnabled(True)
        else:
            self.isolate_button.show()
            self.show_all_button.hide()
            self.isolate_button.setEnabled(selection_exists)
            
    def _create_separator(self):
        """Create a vertical separator line for the toolbar."""
        separator = QLabel("|")
        separator.setStyleSheet("color: gray; margin: 0 5px;")
        return separator

    def on_sort_changed(self, sort_type):
        """Handle sort type change."""
        self.recalculate_widget_positions()

    def _get_sorted_widgets(self):
        """Get widgets sorted according to the current sort setting."""
        sort_type = self.sort_combo.currentText()
        
        if sort_type == "None":
            return list(self.annotation_widgets_by_id.values())
        
        widgets = list(self.annotation_widgets_by_id.values())
        
        if sort_type == "Label":
            widgets.sort(key=lambda w: w.data_item.effective_label.short_label_code)
        elif sort_type == "Image":
            widgets.sort(key=lambda w: os.path.basename(w.data_item.annotation.image_path))
        
        return widgets

    def _group_widgets_by_sort_key(self, widgets):
        """Group widgets by the current sort key and return groups with headers."""
        sort_type = self.sort_combo.currentText()
        
        if sort_type == "None":
            return [("", widgets)]
        
        groups = []
        current_group = []
        current_key = None
        
        for widget in widgets:
            if sort_type == "Label":
                key = widget.data_item.effective_label.short_label_code
            elif sort_type == "Image":
                key = os.path.basename(widget.data_item.annotation.image_path)
            else:
                key = ""
            
            if current_key != key:
                if current_group:
                    groups.append((current_key, current_group))
                current_group = [widget]
                current_key = key
            else:
                current_group.append(widget)
        
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
        
        header = QLabel(text)
        header.setParent(self.content_widget)
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 12px;
                color: #555;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px 8px;
                margin: 2px 0px;
            }
        """)
        header.setFixedHeight(30)  # Increased from 25 to 30
        header.setMinimumWidth(self.viewport().width() - 20)
        header.show()
        
        self._group_headers.append(header)
        return header

    def on_size_changed(self, value):
        """Handle slider value change to resize annotation widgets."""
        if value % 2 != 0:
            value -= 1
        self.current_widget_size = value
        self.size_value_label.setText(str(value))
        
        # Disable updates for performance while resizing many items
        self.content_widget.setUpdatesEnabled(False)
        for widget in self.annotation_widgets_by_id.values():
            widget.update_height(value)  # Call the new, more descriptive method
        self.content_widget.setUpdatesEnabled(True)

        # After resizing, reflow the layout
        self.recalculate_widget_positions()

    def recalculate_grid_layout(self):
        """Recalculate the grid layout based on current widget width."""
        if not self.annotation_widgets_by_id:
            return
            
        available_width = self.viewport().width() - 20
        widget_width = self.current_widget_size + self.grid_layout.spacing()
        cols = max(1, available_width // widget_width)
        
        for i, widget in enumerate(self.annotation_widgets_by_id.values()):
            self.grid_layout.addWidget(widget, i // cols, i % cols)
            
    def recalculate_widget_positions(self):
        """Manually positions widgets in a flow layout with sorting and group headers."""
        if not self.annotation_widgets_by_id:
            self.content_widget.setMinimumSize(1, 1)
            return

        # Clear any existing separator labels
        self._clear_separator_labels()

        # Get sorted widgets
        all_widgets = self._get_sorted_widgets()
        
        # Filter to only visible widgets
        visible_widgets = [w for w in all_widgets if not w.isHidden()]
        
        if not visible_widgets:
            self.content_widget.setMinimumSize(1, 1)
            return

        # Group widgets by sort key
        groups = self._group_widgets_by_sort_key(visible_widgets)

        # Calculate spacing
        spacing = max(5, int(self.current_widget_size * 0.08))
        available_width = self.viewport().width()
        
        x, y = spacing, spacing
        max_height_in_row = 0

        for group_name, group_widgets in groups:
            # Add group header if sorting is enabled and group has a name
            if group_name and self.sort_combo.currentText() != "None":
                # Ensure we're at the start of a new line for headers
                if x > spacing:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0
                
                # Create and position header label
                header_label = self._create_group_header(group_name)
                header_label.move(x, y)
                
                # Move to next line after header
                y += header_label.height() + spacing
                x = spacing
                max_height_in_row = 0

            # Position widgets in this group
            for widget in group_widgets:
                widget_size = widget.size()
                
                # Check if widget fits on current line
                if x > spacing and x + widget_size.width() > available_width:
                    x = spacing
                    y += max_height_in_row + spacing
                    max_height_in_row = 0

                widget.move(x, y)
                x += widget_size.width() + spacing
                max_height_in_row = max(max_height_in_row, widget_size.height())

        # Update content widget size
        total_height = y + max_height_in_row + spacing
        self.content_widget.setMinimumSize(available_width, total_height)
            
    def update_annotations(self, data_items):
        """Update displayed annotations, creating new widgets. This will also
        reset any active isolation view.
        """
        # Reset isolation state before updating to avoid confusion
        if self.isolated_mode:
            self.show_all_annotations()
            
        # Clear any existing widgets and ensure they are deleted
        for widget in self.annotation_widgets_by_id.values():
            widget.setParent(None)
            widget.deleteLater()
            
        self.annotation_widgets_by_id.clear()
        self.selected_widgets.clear()
        self.last_selected_index = -1

        # Create new widgets, parenting them to the content_widget
        for data_item in data_items:
            annotation_widget = AnnotationImageWidget(
                data_item, 
                self.current_widget_size,
                annotation_viewer=self,
                parent=self.content_widget
            )
            annotation_widget.show() 
            self.annotation_widgets_by_id[data_item.annotation.id] = annotation_widget
        
        self.recalculate_widget_positions()
        # Ensure toolbar is in the correct state after a refresh
        self._update_toolbar_state()

    def resizeEvent(self, event):
        """On window resize, reflow the annotation widgets."""
        super().resizeEvent(event)
        # Use a QTimer to avoid rapid, expensive reflows while dragging the resize handle
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self.recalculate_widget_positions)
        # Restart the timer on each resize event
        self._resize_timer.start(100)  # 100ms delay

    def mousePressEvent(self, event):
        """Handle mouse press for starting rubber band selection OR clearing selection."""
        
        # Handle plain left-clicks
        if event.button() == Qt.LeftButton:
            
            # This is the new logic for clearing selection on a background click.
            if not event.modifiers():  # Check for NO modifiers (e.g., Ctrl, Shift)
                
                is_on_widget = False
                child_at_pos = self.childAt(event.pos())

                # Determine if the click was on an actual annotation widget or empty space
                if child_at_pos:
                    widget = child_at_pos
                    while widget and widget != self:
                        if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                            is_on_widget = True
                            break
                        widget = widget.parent()
                
                # If click was on empty space AND something is currently selected...
                if not is_on_widget and self.selected_widgets:
                    # Get IDs of widgets that are about to be deselected to emit a signal
                    changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                    self.clear_selection()
                    self.selection_changed.emit(changed_ids)
                    # The event is handled, but we don't call super() to prevent
                    # the scroll area from doing anything else, like starting a drag.
                    return

            # Handle Ctrl+Click for rubber band
            elif event.modifiers() == Qt.ControlModifier:
                # Store the set of currently selected items.
                self.selection_at_press = set(self.selected_widgets)
                self.rubber_band_origin = event.pos()
                # We determine mouse_pressed_on_widget here but use it in mouseMove
                self.mouse_pressed_on_widget = False
                child_widget = self.childAt(event.pos())
                if child_widget:
                    widget = child_widget
                    while widget and widget != self:
                        if hasattr(widget, 'annotation_viewer') and widget.annotation_viewer == self:
                            self.mouse_pressed_on_widget = True
                            break
                        widget = widget.parent()
                return
                
        # Handle right-clicks
        elif event.button() == Qt.RightButton:
            event.ignore()
            return
            
        # For all other cases (e.g., a click on a widget that should be handled
        # by the widget itself), pass the event to the default handler.
        super().mousePressEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to clear selection and exit isolation mode."""
        if event.button() == Qt.LeftButton:
            changed_ids = []
            
            # If items are selected, clear the selection and record their IDs
            if self.selected_widgets:
                changed_ids = [w.data_item.annotation.id for w in self.selected_widgets]
                self.clear_selection()
                self.selection_changed.emit(changed_ids)

            # If in isolation mode, revert to showing all annotations
            if self.isolated_mode:
                self.show_all_annotations()
            
            # Signal the main window to reset its view (e.g., switch tabs)
            self.reset_view_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for DYNAMIC rubber band selection."""
        if self.rubber_band_origin is None or \
           event.buttons() != Qt.LeftButton or \
           event.modifiers() != Qt.ControlModifier:
            super().mouseMoveEvent(event)
            return

        # If the mouse was pressed on a widget, let that widget handle the event.
        if self.mouse_pressed_on_widget:
            super().mouseMoveEvent(event)
            return

        # Only start the rubber band after dragging a minimum distance
        distance = (event.pos() - self.rubber_band_origin).manhattanLength()
        if distance < self.drag_threshold:
            return

        # Create and show the rubber band if it doesn't exist
        if not self.rubber_band:
            self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.viewport())
        
        rect = QRect(self.rubber_band_origin, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.rubber_band.show()

        # Perform dynamic selection on every move
        selection_rect = self.rubber_band.geometry()
        content_widget = self.content_widget
        changed_ids = []

        for widget in self.annotation_widgets_by_id.values():
            widget_rect_in_content = widget.geometry()
            # Map widget's geometry from the content area to the visible viewport
            widget_rect_in_viewport = QRect(
                content_widget.mapTo(self.viewport(), widget_rect_in_content.topLeft()),
                widget_rect_in_content.size()
            )

            is_in_band = selection_rect.intersects(widget_rect_in_viewport)
            
            # A widget should be selected if it was selected at the start OR is in the band now.
            should_be_selected = (widget in self.selection_at_press) or is_in_band

            if should_be_selected and not widget.is_selected():
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            elif not should_be_selected and widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
        
        if changed_ids:
            self.selection_changed.emit(changed_ids)
        
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete rubber band selection."""
        # Check if a rubber band drag was in progress
        if self.rubber_band_origin is not None and event.button() == Qt.LeftButton:
            if self.rubber_band and self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band.deleteLater()
                self.rubber_band = None

            # **NEEDED CHANGE**: Clean up the stored selection state.
            self.selection_at_press = set()
            self.rubber_band_origin = None
            self.mouse_pressed_on_widget = False
            event.accept()
            return
            
        super().mouseReleaseEvent(event)

    def handle_annotation_selection(self, widget, event):
        """Handle selection of annotation widgets with different modes."""
        # Get the list of widgets to work with based on isolation mode
        if self.isolated_mode:
            # Only work with visible widgets when in isolation mode
            widget_list = [w for w in self.annotation_widgets_by_id.values() if not w.isHidden()]
        else:
            # Use all widgets when not in isolation mode
            widget_list = list(self.annotation_widgets_by_id.values())
        
        try:
            widget_index = widget_list.index(widget)
        except ValueError:
            return

        modifiers = event.modifiers()
        changed_ids = []

        # --- The selection logic now identifies which items to change   ---
        # --- but the core state change happens in select/deselect       ---
        
        if modifiers == Qt.ShiftModifier or modifiers == (Qt.ShiftModifier | Qt.ControlModifier):
            # Range selection
            if self.last_selected_index != -1:
                # Find the last selected widget in the current widget list
                last_selected_widget = None
                for w in self.selected_widgets:
                    if w in widget_list:
                        try:
                            last_index_in_current_list = widget_list.index(w)
                            if last_selected_widget is None or \
                               last_index_in_current_list > widget_list.index(last_selected_widget):
                                last_selected_widget = w
                        except ValueError:
                            continue
                
                if last_selected_widget:
                    last_selected_index_in_current_list = widget_list.index(last_selected_widget)
                    start = min(last_selected_index_in_current_list, widget_index)
                    end = max(last_selected_index_in_current_list, widget_index)
                else:
                    # Fallback if no previously selected widget is found in current list
                    start = widget_index
                    end = widget_index
                
                for i in range(start, end + 1):
                    # select_widget will return True if a change occurred
                    if self.select_widget(widget_list[i]):
                        changed_ids.append(widget_list[i].data_item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
                self.last_selected_index = widget_index
        
        elif modifiers == Qt.ControlModifier:
            # Toggle selection
            if widget.is_selected():
                if self.deselect_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            else:
                if self.select_widget(widget):
                    changed_ids.append(widget.data_item.annotation.id)
            self.last_selected_index = widget_index
                
        else:
            # Normal click: clear all others and select this one
            newly_selected_id = widget.data_item.annotation.id
            # Deselect all widgets that are not the clicked one
            for w in list(self.selected_widgets):
                if w.data_item.annotation.id != newly_selected_id:
                    if self.deselect_widget(w):
                        changed_ids.append(w.data_item.annotation.id)
            # Select the clicked widget
            if self.select_widget(widget):
                changed_ids.append(newly_selected_id)
            self.last_selected_index = widget_index
            
        # Update isolation if in isolated mode
        if self.isolated_mode:
            self._update_isolation()
        
        # If any selections were changed, emit the signal
        if changed_ids:
            self.selection_changed.emit(changed_ids)
            
    def _update_isolation(self):
        """Update the isolated view to show only currently selected widgets."""
        if not self.isolated_mode:
            return
            
        if self.selected_widgets:
            # ADD TO isolation instead of replacing it
            self.isolated_widgets.update(self.selected_widgets)  # Use update() to add, not replace
            self.setUpdatesEnabled(False)
            try:
                for widget in self.annotation_widgets_by_id.values():
                    if widget not in self.isolated_widgets:
                        widget.hide()
                    else:
                        widget.show()
                self.recalculate_widget_positions()
            finally:
                self.setUpdatesEnabled(True)
        else:
            # If no widgets are selected, keep the current isolation (don't exit)
            # This prevents accidentally exiting isolation mode when clearing selection
            pass

    def select_widget(self, widget):
        """Select a widget, update the data_item, and return True if state changed."""
        if not widget.is_selected():
            widget.set_selected(True)
            widget.data_item.set_selected(True)
            self.selected_widgets.append(widget)
            self.update_label_window_selection()
            self._update_toolbar_state()  # Update button states
            return True
        return False

    def deselect_widget(self, widget):
        """Deselect a widget, update the data_item, and return True if state changed."""
        if widget.is_selected():
            widget.set_selected(False)
            widget.data_item.set_selected(False)
            if widget in self.selected_widgets:
                self.selected_widgets.remove(widget)
            self.update_label_window_selection()
            self._update_toolbar_state()  # Update button states
            return True
        return False

    def clear_selection(self):
        """Clear all selected widgets and update toolbar state."""
        for widget in list(self.selected_widgets):
            widget.set_selected(False)
        self.selected_widgets.clear()
        self.update_label_window_selection()
        self._update_toolbar_state()  # Update button states

    def update_label_window_selection(self):
        """Update the label window selection based on currently selected annotations."""
        explorer_window = self.parent()
        while explorer_window and not hasattr(explorer_window, 'main_window'):
            explorer_window = explorer_window.parent()
            
        if not explorer_window or not hasattr(explorer_window, 'main_window'):
            return
            
        main_window = explorer_window.main_window
        label_window = main_window.label_window
        annotation_window = main_window.annotation_window
        
        if not self.selected_widgets:
            label_window.deselect_active_label()
            label_window.update_annotation_count()
            return
            
        selected_data_items = [widget.data_item for widget in self.selected_widgets]
        
        first_effective_label = selected_data_items[0].effective_label
        all_same_current_label = True
        for item in selected_data_items:
            if item.effective_label.id != first_effective_label.id:
                all_same_current_label = False
                break
        
        if all_same_current_label:
            label_window.set_active_label(first_effective_label)
            if not selected_data_items[0].has_preview_changes():
                annotation_window.labelSelected.emit(first_effective_label.id)
        else:
            label_window.deselect_active_label()
        
        label_window.update_annotation_count()

    def get_selected_annotations(self):
        """Get the annotations corresponding to selected widgets."""
        return [widget.annotation for widget in self.selected_widgets]
    
    def render_selection_from_ids(self, selected_ids):
        """Update the visual selection of widgets based on a set of IDs from the controller."""
        # Block signals temporarily to prevent cascade updates
        self.setUpdatesEnabled(False)
        
        try:
            for ann_id, widget in self.annotation_widgets_by_id.items():
                is_selected = ann_id in selected_ids
                widget.set_selected(is_selected)
            
            # Resync internal list of selected widgets
            self.selected_widgets = [w for w in self.annotation_widgets_by_id.values() if w.is_selected()]
            
            # If we're in isolated mode, ADD to the isolation instead of replacing it
            if self.isolated_mode and self.selected_widgets:
                self.isolated_widgets.update(self.selected_widgets)  # Add to existing isolation
                # Hide all widgets except those in the isolated set
                for widget in self.annotation_widgets_by_id.values():
                    if widget not in self.isolated_widgets:
                        widget.hide()
                    else:
                        widget.show()
                self.recalculate_widget_positions()
            
        finally:
            self.setUpdatesEnabled(True)
        
        # Update label window once at the end
        self.update_label_window_selection()
        # Update toolbar state to enable/disable Isolate button
        self._update_toolbar_state()
    
    def apply_preview_label_to_selected(self, preview_label):
        """Apply a preview label and emit a signal for the embedding view to update."""
        if not self.selected_widgets or not preview_label:
            return

        changed_ids = []
        for widget in self.selected_widgets:
            widget.data_item.set_preview_label(preview_label)
            widget.update() # Force repaint with new color
            changed_ids.append(widget.data_item.annotation.id)
            
        # Recalculate positions to update sorting based on new effective labels
        if self.sort_combo.currentText() == "Label":
            self.recalculate_widget_positions()

        if changed_ids:
            self.preview_changed.emit(changed_ids)

    def clear_preview_states(self):
        """Clear all preview states and revert to original labels."""
        # We just need to iterate through all widgets and tell their data_items to clear
        something_cleared = False
        for widget in self.annotation_widgets_by_id.values():
            if widget.data_item.has_preview_changes():
                widget.data_item.clear_preview_label()
                widget.update() # Repaint to show original color
                something_cleared = True
        
        if something_cleared:
            # Recalculate positions to update sorting based on reverted labels
            if self.sort_combo.currentText() == "Label":
                self.recalculate_widget_positions()
            self.update_label_window_selection()

    def has_preview_changes(self):
        """Check if there are any pending preview changes."""
        return any(w.data_item.has_preview_changes() for w in self.annotation_widgets_by_id.values())

    def get_preview_changes_summary(self):
        """Get a summary of preview changes for user feedback."""
        change_count = sum(1 for w in self.annotation_widgets_by_id.values() if w.data_item.has_preview_changes())
        if not change_count:
            return "No preview changes"
        return f"{change_count} annotation(s) with preview changes"

    def apply_preview_changes_permanently(self):
        """Apply all preview changes permanently to the annotation data."""
        applied_annotations = []
        for widget in self.annotation_widgets_by_id.values():
            # Tell the data_item to apply its changes to the underlying annotation
            if widget.data_item.apply_preview_permanently():
                applied_annotations.append(widget.annotation)
        
        return applied_annotations