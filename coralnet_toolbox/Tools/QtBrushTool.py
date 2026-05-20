import warnings
import numpy as np

from PyQt5.QtGui import QColor, QPen, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QObject, pyqtSignal, QRunnable, QThreadPool
from PyQt5.QtWidgets import QGraphicsEllipseItem, QMessageBox, QGraphicsRectItem, QGraphicsPathItem

from coralnet_toolbox.QtActions import MaskEditAction
from coralnet_toolbox.Tools.QtTool import Tool

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------------------------------------------------------------------------------------------------
# Background Threading
# ----------------------------------------------------------------------------------------------------------------------

class StrokeMathSignals(QObject):
    # Emits: flat_indices, center_pos, combined_mask, mask_annotation_ref, label_id_str
    finished = pyqtSignal(object, object, object, object, str)

class StrokeMathWorker(QRunnable):
    def __init__(self, points, brush_size, brush_mask, img_w, img_h, mask_annotation, label_id, z_channel=None):
        super().__init__()

        self.points = points
        self.brush_size = brush_size
        self.brush_mask = brush_mask
        self.img_w = img_w
        self.img_h = img_h
        self.mask_annotation = mask_annotation
        self.label_id = label_id
        self.z_channel = z_channel

        self.signals = StrokeMathSignals()

    def run(self):
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        
        min_x = int(min(xs) - self.brush_size / 2.0)
        max_x = int(max(xs) + self.brush_size / 2.0)
        min_y = int(min(ys) - self.brush_size / 2.0)
        max_y = int(max(ys) + self.brush_size / 2.0)
        
        w, h = max_x - min_x, max_y - min_y
        combined_mask = np.zeros((h, w), dtype=bool)
        
        for p in self.points:
            px_local = int(p.x() - self.brush_size / 2.0) - min_x
            py_local = int(p.y() - self.brush_size / 2.0) - min_y
            
            bh, bw = self.brush_mask.shape
            if (0 <= px_local < w and 0 <= py_local < h and
                px_local + bw > 0 and py_local + bh > 0):
                
                ystart = max(0, py_local)
                yend = min(h, py_local + bh)
                xstart = max(0, px_local)
                xend = min(w, px_local + bw)
                
                brush_ystart = ystart - py_local
                brush_yend = brush_ystart + (yend - ystart)
                brush_xstart = xstart - px_local
                brush_xend = brush_xstart + (xend - xstart)
                
                brush_clip = self.brush_mask[brush_ystart:brush_yend, brush_xstart:brush_xend]
                
                # --- STATISTICAL DEPTH FILTER ---
                if self.z_channel is not None:
                    # Translate local chunk coordinates to global image coordinates
                    global_ystart = min_y + ystart
                    global_yend = min_y + yend
                    global_xstart = min_x + xstart
                    global_xend = min_x + xend
                    
                    # Ensure we don't slice outside the z_channel bounds
                    if (global_ystart >= 0 and global_yend <= self.img_h and 
                        global_xstart >= 0 and global_xend <= self.img_w):
                        
                        depth_slice = self.z_channel[global_ystart:global_yend, global_xstart:global_xend]
                        
                        # 1. THE PINPOINT ANCHOR (Strict 3x3 core)
                        cy = (yend - ystart) // 2
                        cx = (xend - xstart) // 2
                        
                        core_y_start = max(0, cy - 1)
                        core_y_end = min(depth_slice.shape[0], cy + 2)
                        core_x_start = max(0, cx - 1)
                        core_x_end = min(depth_slice.shape[1], cx + 2)
                        
                        core_depths = depth_slice[core_y_start:core_y_end, core_x_start:core_x_end]
                        core_depths = core_depths[~np.isnan(core_depths)]
                        
                        if len(core_depths) == 0:
                            valid_depths = depth_slice[brush_clip]
                            core_depths = valid_depths[~np.isnan(valid_depths)]
                            
                        if len(core_depths) > 0:
                            median_z = np.median(core_depths)
                            
                            # 2. RAZOR-THIN TOLERANCE (1.5% of depth, min 2cm)
                            tolerance = max(0.02, median_z * 0.015) 
                            
                            # 3. BRICK WALL CLIFF PROTECTION
                            with np.errstate(invalid='ignore'):
                                # Allow 1x tolerance for things sticking out (closer to camera, -Z)
                                # Allow ONLY 0.5x tolerance for things falling away (further from camera, +Z)
                                depth_mask = (depth_slice >= (median_z - tolerance)) & (depth_slice <= (median_z + tolerance * 0.5))
                                
                            brush_clip = brush_clip & depth_mask

                # ---------------------------------------------------------
                
                combined_mask[ystart:yend, xstart:xend] |= brush_clip

        local_ys, local_xs = np.where(combined_mask)
        global_xs = local_xs + min_x
        global_ys = local_ys + min_y
        
        valid = (global_xs >= 0) & (global_xs < self.img_w) & (global_ys >= 0) & (global_ys < self.img_h)
        flat_indices = (global_ys[valid] * self.img_w + global_xs[valid]).astype(np.int64)

        center_pos = QPointF(min_x + w / 2.0, min_y + h / 2.0)
        
        self.signals.finished.emit(flat_indices, center_pos, combined_mask, self.mask_annotation, self.label_id)

# ----------------------------------------------------------------------------------------------------------------------
# Tool Class
# ----------------------------------------------------------------------------------------------------------------------

class BrushTool(Tool):
    """A tool for painting on a MaskAnnotation layer."""
    def __init__(self, annotation_window):
        super().__init__(annotation_window)
        
        self.show_crosshair = False
        self.cursor = Qt.CrossCursor 
        
        self.brush_size = 90
        self.shape = 'circle'
        self.brush_mask = self._create_brush_mask()
        self.painting = False

        self.post_stroke_callback = None
        self._accumulated_points = []
        self._stroke_history_action = None
        self._stroke_mask_annotation = None
        self._last_scratchpad_pos = None
        self._stroke_accumulated_indices = []
        
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
        
        # LIVE STREAMING ENGINE
        self._sync_timer = QTimer()
        self._sync_timer.timeout.connect(self._stream_stroke_chunk)
        self._is_finishing_stroke = False
        self._active_workers = 0

    def _create_brush_mask(self):
        if self.shape == 'circle':
            radius = self.brush_size // 2
            y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
            return x**2 + y**2 <= radius**2
        elif self.shape == 'square':
            size = self.brush_size
            return np.ones((size, size), dtype=bool)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
            
        if not self.annotation_window.selected_label:
            QMessageBox.warning(self.annotation_window,
                                "No Label Selected",
                                "A label must be selected before using the brush tool.")
            return
        
        if not self.annotation_window.cursorInWindow(event.pos()):
            return

        if not self.annotation_window.selected_label.is_visible:
            self.annotation_window.selected_label.visibility_checkbox.setChecked(True)

        if not self.painting and self._stroke_history_action is not None:
            return

        self.painting = not self.painting
        
        if self.painting:
            # 1. Setup the Qt Scratchpad (Thick Polyline)
            self._is_finishing_stroke = False
            self._stroke_accumulated_indices.clear()
            self._last_scratchpad_pos = None

            self._stroke_mask_annotation = self.annotation_window.current_mask_annotation
            self._stroke_history_action = MaskEditAction(
                self._stroke_mask_annotation,
                description=f"{self.__class__.__name__} stroke",
            )
            self.scratchpad_path = QPainterPath()
            self.scratchpad_item = QGraphicsPathItem()
            
            label = self.annotation_window.selected_label
            transparency = self.annotation_window.main_window.get_transparency_value()
            
            c = QColor(label.color)
            c.setAlpha(transparency)

            # Use a thick Pen instead of a filled polygon
            pen = QPen(c)
            pen.setWidth(self.brush_size)
            if self.shape == 'circle':
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
            else:
                pen.setCapStyle(Qt.SquareCap)
                pen.setJoinStyle(Qt.BevelJoin)

            self.scratchpad_item.setPen(pen)
            self.scratchpad_item.setBrush(QBrush(Qt.NoBrush))
            self.scratchpad_item.setZValue(3)
            
            self.annotation_window.scene.addItem(self.scratchpad_item)
            
            # 2. Start streaming chunks at 25 FPS
            self._sync_timer.start(40)
            
            self._apply_brush(event)
        else:
            self._finish_stroke()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        
        if (cursor_in_window and self.active and self.annotation_window.selected_label):
            self.update_cursor_annotation(scene_pos)
            if self.cursor_move_callback:
                self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)
        else:
            self.clear_cursor_annotation()
            if self.cursor_clear_callback:
                self.cursor_clear_callback()

        if self.painting:
            self._apply_brush(event)
    
    def keyPressEvent(self, event):
        modifiers = event.modifiers()
        if ((modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier)) and self.active:
            self._toggle_shape()
        super().keyPressEvent(event)

    def _toggle_shape(self):
        self.shape = 'square' if self.shape == 'circle' else 'circle'
        self.brush_mask = self._create_brush_mask()
        cursor_pos = self.annotation_window.mapFromGlobal(self.annotation_window.cursor().pos())
        scene_pos = self.annotation_window.mapToScene(cursor_pos)

        if self.cursor_annotation:
            self.update_cursor_annotation(scene_pos)

        manager = getattr(self.main_window, 'mvat_manager', None)
        if manager is not None:
            try:
                manager.on_2d_tool_size_changed(self, scene_pos)
            except Exception:
                pass

    def set_brush_size(self, size, propagate: bool = True):
        """Set brush diameter (in image pixels) and mirror it to the sibling tool.

        Brush and Erase are separate Tool instances but the user expects them
        to share a size — switching between them should keep the same brush
        footprint. When propagate=True we copy the new size onto whichever
        sibling tool (brush↔erase) lives on the same AnnotationWindow.
        """
        try:
            new_size = max(1, int(round(float(size))))
        except Exception:
            return

        if new_size == self.brush_size:
            return

        self.brush_size = new_size
        try:
            self.brush_mask = self._create_brush_mask()
        except Exception:
            pass

        if not propagate:
            return

        # Mirror onto the sibling tool (brush ↔ erase) so size is shared.
        try:
            tools = getattr(self.annotation_window, 'tools', {})
        except Exception:
            tools = {}
        sibling_name = 'erase' if type(self).__name__ == 'BrushTool' else 'brush'
        sibling = tools.get(sibling_name)
        if sibling is not None and sibling is not self:
            try:
                sibling.set_brush_size(new_size, propagate=False)
            except Exception:
                # Fallback: directly poke the attributes if the sibling
                # doesn't expose the setter for any reason.
                try:
                    sibling.brush_size = new_size
                    sibling.brush_mask = sibling._create_brush_mask()
                except Exception:
                    pass

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()

            # Multiplicative resize so the step feels the same regardless of
            # image resolution. A fixed +/-5 px step is barely visible on a
            # multi-thousand-pixel raster (which is why users were complaining
            # the 2D brush size "doesn't change"). Mirror the 3D tool's 15%
            # per-notch behaviour: positive notch grows, negative shrinks.
            notches = (delta / 120.0) if delta else (1.0 if delta >= 0 else -1.0)
            factor = 1.15 ** notches

            current = max(1, int(self.brush_size))
            target = current * factor
            # Ensure each notch actually changes the integer pixel size so
            # tiny brushes (e.g. 1–6 px) still respond to the wheel.
            if abs(target - current) < 1.0:
                target = current + (1.0 if notches > 0 else -1.0)
            self.set_brush_size(max(1, int(round(target))))

            scene_pos = self.annotation_window.mapToScene(event.pos())
            self.update_cursor_annotation(scene_pos)

            manager = getattr(self.main_window, 'mvat_manager', None)
            if manager is not None:
                try:
                    manager.on_2d_tool_size_changed(self, scene_pos)
                except Exception:
                    pass

    def create_cursor_preview_item(self, u: float, v: float, radius: float = None):
        if not self.annotation_window.selected_label:
            return None
        
        label = self.annotation_window.selected_label
        transparency = self.annotation_window.main_window.get_transparency_value()
        radius = self.brush_size / 2.0 if radius is None else float(radius)
        diameter = max(1.0, radius * 2.0)
        c = QColor(label.color)
        fill = QColor(c)
        fill.setAlpha(transparency)
        pen = QPen(c.darker(150), 2)
        pen.setCosmetic(True)
        
        if self.shape == 'circle':
            item = QGraphicsEllipseItem(u - radius, v - radius, diameter, diameter)
        else:
            item = QGraphicsRectItem(u - radius, v - radius, diameter, diameter)
        
        item.setBrush(QBrush(fill))
        item.setPen(pen)
        return item

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        if (not scene_pos or not self.annotation_window.selected_label or 
            not self.annotation_window.active_image or 
            not self.annotation_window.main_window.label_window.active_label):
            self.clear_cursor_annotation()
            return
            
        self.clear_cursor_annotation()
        
        label_color = self.annotation_window.selected_label.color
        transparency = self.annotation_window.main_window.get_transparency_value()
        
        radius = self.brush_size / 2.0
        rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        
        if self.shape == 'circle':
            self.cursor_annotation = QGraphicsEllipseItem(rect)
        else:
            self.cursor_annotation = QGraphicsRectItem(rect)
        
        brush_color = QColor(label_color)
        brush_color.setAlpha(transparency)
        self.cursor_annotation.setBrush(brush_color)
        
        border_color = QColor(label_color).darker(150)
        pen = QPen(border_color)
        pen.setWidth(2)
        self.cursor_annotation.setPen(pen)
        
        self.annotation_window.scene.addItem(self.cursor_annotation)
        
        self._last_brush_size = self.brush_size
        self._last_shape = self.shape

    def update_cursor_annotation(self, scene_pos: QPointF = None):
        if scene_pos is None:
            self.clear_cursor_annotation()
            return
        
        if (self.cursor_annotation and 
            hasattr(self, '_last_brush_size') and self._last_brush_size == self.brush_size and
            hasattr(self, '_last_shape') and self._last_shape == self.shape):
            radius = self.brush_size / 2.0
            rect = QRectF(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
            self.cursor_annotation.setRect(rect)
        else:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)

    def clear_cursor_annotation(self):
        if self.cursor_annotation and self.cursor_annotation.scene():
            self.annotation_window.scene.removeItem(self.cursor_annotation)
            self.cursor_annotation = None

    def deactivate(self):
        if self.painting:
            self._finish_stroke()
        super().deactivate()
        
    def stop_current_drawing(self):
        if self.painting:
            self._finish_stroke()

    def _apply_brush(self, event):
        """Draws visually on the Qt Scratchpad and accumulates points (Zero NumPy)."""
        scene_pos = self.annotation_window.mapToScene(event.pos())
        
        if self.scratchpad_item:
            if self._last_scratchpad_pos is None:
                self.scratchpad_path.moveTo(scene_pos)
                # Draw a tiny micro-line to itself to ensure a single click renders a dot
                self.scratchpad_path.lineTo(scene_pos.x() + 0.1, scene_pos.y())
            else:
                self.scratchpad_path.lineTo(scene_pos)

            self.scratchpad_item.setPath(self.scratchpad_path)
            self._last_scratchpad_pos = scene_pos

        self._accumulated_points.append(scene_pos)

    def _stream_stroke_chunk(self):
        """Grabs the recent points and sends them to the background worker."""
        if not self._accumulated_points:
            # Safely cleanup if we are waiting to finish and no workers are running
            if self._is_finishing_stroke and self._active_workers == 0:
                self._cleanup_scratchpad()
                self.annotation_window.viewport().update()
                self._commit_stroke_history_action()
            return
            
        points_to_process = list(self._accumulated_points)
        self._accumulated_points.clear()
        
        mask_annotation = self._stroke_mask_annotation or self.annotation_window.current_mask_annotation
        if not mask_annotation or not self.annotation_window.selected_label:
            if self._is_finishing_stroke and self._active_workers == 0:
                self._cleanup_scratchpad()
                self._commit_stroke_history_action()
            return

        img_h, img_w = mask_annotation.mask_data.shape
        
        # Safely get the Z-channel
        raster = self.annotation_window.main_window.image_window.raster_manager.get_raster(mask_annotation.image_path)
        z_channel = raster.z_channel if raster else None
        
        self._active_workers += 1
        worker = StrokeMathWorker(
            points=points_to_process, 
            brush_size=self.brush_size, 
            brush_mask=self.brush_mask, 
            img_w=img_w, 
            img_h=img_h,
            mask_annotation=mask_annotation,
            label_id=self.annotation_window.selected_label.id,
            z_channel=z_channel
        )
        worker.signals.finished.connect(self._on_math_finished)
        QThreadPool.globalInstance().start(worker)

    def _finish_stroke(self):
        """Called when the user lets go of the mouse to finalize the stroke."""
        self.painting = False
        self._sync_timer.stop()
        self._is_finishing_stroke = True
        
        # Flush any remaining points immediately
        self._stream_stroke_chunk()

    def _on_math_finished(self, flat_indices, center_pos, combined_mask, mask_annotation, selected_label_id):
        """Executes on the Main Thread: Writes the arrays locally and defers 3D sync."""
        self._active_workers -= 1
        
        class_id = mask_annotation.label_id_to_class_id_map.get(selected_label_id)
        if class_id is not None and len(flat_indices) > 0:
            # Update the active canvas instantly
            mask_annotation.update_mask_at_indices(
                flat_indices,
                class_id,
                silent=True,
                history_action=self._stroke_history_action,
            )
            # Accumulate the flat indices for deferred global propagation
            self._stroke_accumulated_indices.append(flat_indices)

        # CLEANUP & DEFERRED GLOBAL PROPAGATION
        # Only trigger the heavy 3D math when the user has released the stroke
        # AND all background workers have finished computing the arrays.
        if self._is_finishing_stroke and self._active_workers == 0:

            # Did we actually paint anything?
            if self._stroke_accumulated_indices and self.post_stroke_callback:
                # 1. Flatten all painted pixels across the entire stroke into one array
                combined_flat = np.unique(np.concatenate(self._stroke_accumulated_indices))

                if len(combined_flat) > 0:
                    h, w = mask_annotation.mask_data.shape

                    # 2. Find the tight bounding box of the entire stroke
                    y_coords, x_coords = np.divmod(combined_flat, w)
                    min_x, max_x = int(x_coords.min()), int(x_coords.max())
                    min_y, max_y = int(y_coords.min()), int(y_coords.max())

                    crop_w = (max_x - min_x) + 1
                    crop_h = (max_y - min_y) + 1

                    # 3. Create a compact boolean mask of just the painted area
                    cropped_mask = np.zeros((crop_h, crop_w), dtype=bool)
                    local_y = y_coords - min_y
                    local_x = x_coords - min_x
                    cropped_mask[local_y, local_x] = True

                    # 4. Fire ONE heavy payload to the MVAT Manager
                    final_center = QPointF(min_x + crop_w / 2.0, min_y + crop_h / 2.0)
                    self.post_stroke_callback(final_center, selected_label_id, cropped_mask)

            # Final Cleanup
            self._cleanup_scratchpad()
            self._last_scratchpad_pos = None
            self._stroke_accumulated_indices.clear()
            # Final local repaint so the baked pixels show up
            self.annotation_window.viewport().update()
            self._commit_stroke_history_action()

    def _cleanup_scratchpad(self):
        """Safely removes the temporary vector stroke from the UI."""
        if self.scratchpad_item and self.scratchpad_item.scene():
            self.annotation_window.scene.removeItem(self.scratchpad_item)
        self.scratchpad_item = None
        self.scratchpad_path = QPainterPath()
        self._accumulated_points.clear()

    def _commit_stroke_history_action(self):
        mask_annotation = self._stroke_mask_annotation
        if self._stroke_history_action and not self._stroke_history_action.is_empty():
            self.annotation_window.action_stack.push(self._stroke_history_action)
            if mask_annotation is not None:
                try:
                    mask_annotation.annotationUpdated.emit(mask_annotation)
                except Exception:
                    pass
        self._stroke_history_action = None
        self._stroke_mask_annotation = None
        self._is_finishing_stroke = False