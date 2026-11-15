from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QMouseEvent, QColor, QPen, QCursor
from PyQt5.QtWidgets import QGraphicsLineItem

from coralnet_toolbox.QtActions import PromptStateAction, WorkAreaAction


# Z for transient tool graphics on the main scene (cut line, brush and eraser
# cursors, box select). Anything left at the default 0 is drawn beneath the
# phantom annotation layer (10) and selected annotations (20), so it vanishes
# exactly where the user is working. Level with the crosshair.
TOOL_OVERLAY_Z = 1000


def fine_dash_pen(color, width: float = 2.0) -> QPen:
    """Cosmetic fine dashed pen for tool guide lines (cut line, scale line).

    Cosmetic, so it stays the same on screen at any zoom. Dash pattern units
    are multiples of the pen width: at width 2 that is a 6 px dash, 4 px gap.
    """
    pen = QPen(QColor(color), width)
    pen.setDashPattern([3.0, 2.0])
    pen.setCosmetic(True)
    return pen


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Tool:
    # A tool's hint is the status bar's resting state, so it is posted with no
    # timeout. Anything else a tool or an operation shows covers it, and
    # MainWindow puts it back as soon as that message expires.
    STATUS_HINT_MSECS = 0

    def __init__(self, annotation_window):
        self.annotation_window = annotation_window
        self.main_window = annotation_window.main_window

        self.active = False
        self.cursor = Qt.ArrowCursor
        self.default_cursor = Qt.ArrowCursor
        self.cursor_annotation = None

        # Optional callbacks for cursor preview propagation (set by any consumer).
        # Subclasses that support live cursor previews should call these in their mouseMoveEvent.
        # cursor_move_callback(scene_pos: QPointF, item_factory: callable)
        #   item_factory(u: float, v: float) -> QGraphicsItem  (positioned at image coords u,v)
        self.cursor_move_callback = None
        # cursor_clear_callback()
        self.cursor_clear_callback = None

        # Debounce timer — throttle cursor-preview projection to ~10 Hz.
        # Subclasses call _schedule_cursor_preview_update(scene_pos) in mouseMoveEvent;
        # the base deactivate() and _fire_cursor_preview_update() handle the rest.
        self._cursor_update_timer = QTimer()
        self._cursor_update_timer.setSingleShot(True)
        self._cursor_update_timer.timeout.connect(self._fire_cursor_preview_update)
        self._pending_cursor_pos = None

        # Last hover position in scene coords, recorded by mouseMoveEvent.
        # refresh_label_preview() needs it to rebuild previews in place when the
        # selected label changes without the mouse moving.
        self.last_scene_pos = None

        # A tool's own undo/redo stack, for steps that are not annotations yet:
        # prompts, a preview, a work area. Tools with none leave it None and the
        # hotkey falls through to the annotation window's stack (see tool_undo).
        self.tool_stack = None

        # Crosshair settings
        self.show_crosshair = True  # Flag to toggle crosshair visibility for this tool
        self.h_crosshair_line = None
        self.v_crosshair_line = None

    def activate(self):
        self.active = True
        self.annotation_window.setCursor(self.cursor)
        self.report_state()

    def status_hint(self):
        """Say what the mouse and keys do from the tool's current state.

        Returns one line in the form "Action: effect  |  Action: effect", or
        None to leave the status bar alone. Subclasses override this; the
        toolbar tooltip holds the full reference, this holds what applies now.
        """
        return None

    def report_state(self):
        """Put status_hint() on the status bar and leave it there.

        Called on activation and wherever the tool changes state, so the hint
        always describes what the next click or key press will do. It stays up
        for as long as the tool is active: MainWindow watches the status bar and
        reposts the hint whenever a timed message that covered it expires, so a
        tool's instructions are what the user sees whenever nothing more
        pressing is being said.

        Callers that just showed an outcome message (an error, a count) should
        still not call this straight after, or the hint replaces it at once
        rather than after the user has read it.
        """
        if not self.active:
            return
        message = self.status_hint()
        if not message:
            return
        try:
            self.main_window.show_tool_hint(message)
        except AttributeError:
            # A main window without the hint plumbing (tests, embedding).
            try:
                self.main_window.status_bar.showMessage(message, self.STATUS_HINT_MSECS)
            except Exception:
                pass
        except Exception:
            pass

    def deactivate(self):
        self._cursor_update_timer.stop()
        self._pending_cursor_pos = None
        self.last_scene_pos = None
        self.active = False
        self.annotation_window.setCursor(self.default_cursor)
        self.clear_cursor_annotation()
        if self.cursor_clear_callback:
            self.cursor_clear_callback()

        # Ensure crosshair is properly cleared when deactivating tool
        self.clear_crosshair()

        # The hint has no timeout, so it would outlive the tool that posted it.
        try:
            self.main_window.clear_tool_hint()
        except Exception:
            pass

        # Stop any current drawing operation
        self.stop_current_drawing()

    def leave(self):
        """Drop hover-only graphics because the pointer left the window.

        Qt stops delivering mouseMoveEvent the instant the pointer exits the
        widget, so every clear that lives inside mouseMoveEvent never runs on
        the way out and the graphics stay on screen. AnnotationWindow.leaveEvent
        calls this instead.

        Deliberately does NOT call stop_current_drawing(): a toggled-on brush
        stroke, a half-drawn polygon, or a pending SAM work area must survive
        the pointer leaving. Subclasses that own extra *hover-driven* state
        (debounce timers, temp predictions) should override, clear that, and
        call super().leave().
        """
        self._cursor_update_timer.stop()
        self._pending_cursor_pos = None
        self.last_scene_pos = None
        self.clear_cursor_annotation()
        self.clear_crosshair()
        if self.cursor_clear_callback:
            self.cursor_clear_callback()

    def _pointer_over_window(self) -> bool:
        """True while the pointer is physically over the annotation viewport.
        Used to stop debounce timers from re-adding hover graphics after a leave.
        """
        try:
            viewport = self.annotation_window.viewport()
            return viewport.rect().contains(viewport.mapFromGlobal(QCursor.pos()))
        except Exception:
            return False

    def mousePressEvent(self, event: QMouseEvent):
        pass

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Base implementation of mouseMoveEvent that handles crosshair display.
        Child classes should call super().mouseMoveEvent(event) in their implementation.
        """
        # Handle crosshair display
        scene_pos = self.annotation_window.mapToScene(event.pos())
        cursor_in_window = self.annotation_window.cursorInWindow(event.pos())
        self.last_scene_pos = scene_pos if cursor_in_window else None

        if (cursor_in_window and self.active and 
            self.annotation_window.selected_label and 
            self.show_crosshair):
            self.update_crosshair(scene_pos)
        else:
            self.clear_crosshair()

    def mouseReleaseEvent(self, event: QMouseEvent):
        pass

    def keyPressEvent(self, event):
        pass

    def keyReleaseEvent(self, event):
        pass

    def wheelEvent(self, event: QMouseEvent):
        pass
        
    def _schedule_cursor_preview_update(self, scene_pos: QPointF):
        """Queue a debounced cursor-preview update (call from mouseMoveEvent when a consumer is registered)."""
        self._pending_cursor_pos = scene_pos
        if not self._cursor_update_timer.isActive():
            self._cursor_update_timer.start(100)  # ~10 Hz throttle

    def _fire_cursor_preview_update(self):
        """Timer callback — fire cursor_move_callback if a consumer is registered."""
        # The pointer can leave during the 100 ms debounce; firing anyway would
        # re-add the context-canvas previews leave() just removed.
        if not self._pointer_over_window():
            self._pending_cursor_pos = None
            return
        
        if self._pending_cursor_pos is not None and self.cursor_move_callback:
            self.cursor_move_callback(self._pending_cursor_pos, self.create_cursor_preview_item)

    def refresh_label_preview(self):
        """Rebuild live preview graphics after the selected label changed.

        AnnotationWindow.set_selected_label calls this on the active tool. It
        fires whether or not any annotation is selected, because the previews it
        repaints — the cursor annotation, the projected preview on context
        canvases — exist purely from hovering and would otherwise keep showing
        the previous label's color until the mouse moved again.

        Base behaviour covers every tool whose preview is just a cursor
        annotation built from annotation_window.selected_label. Tools holding
        extra label-colored graphics (prompt rectangles, unconfirmed
        predictions) override this.
        """
        if not self.active:
            return

        scene_pos = self.last_scene_pos
        if scene_pos is None or not self._pointer_over_window():
            # Nothing is being hovered, so there is no preview to recolor.
            self.clear_cursor_annotation()
            if self.cursor_clear_callback:
                self.cursor_clear_callback()
            return

        self.clear_cursor_annotation()
        self.create_cursor_annotation(scene_pos)

        # Push the rebuilt preview out to any consumer canvases as well.
        if self.cursor_move_callback:
            self.cursor_move_callback(scene_pos, self.create_cursor_preview_item)

    def stop_current_drawing(self):
        """
        Force stop of the current drawing operation if one is in progress.
        Subclasses should override this to implement tool-specific stopping logic.
        """
        pass

    def clear_image(self):
        """Drop whatever image this tool holds for its work area.

        Called wherever a work area is torn down. A tool that encodes a crop of
        the raster holds a viewport-sized array for as long as it is selected
        otherwise, and -- worse -- keeps reporting that there is something to
        predict against after the work area it belonged to is gone. Tools that
        hold no image inherit the no-op.
        """
        pass

    def prompt_state(self):
        """A snapshot of the prompts this tool is holding, or None.

        Tools with their own stack override this and restore_prompt_state as a
        pair; the two are what _record_prompt_change records between.
        """
        return None

    def restore_prompt_state(self, state):
        """Put prompts back as a snapshot had them. Overridden with prompt_state."""
        pass

    def _record_prompt_change(self, before, description):
        """Record a prompt change on this tool's stack, if anything changed.

        Called with a snapshot taken before the change and a phrase naming it.
        Nothing is recorded when the snapshot is identical, so a click that
        missed, or a clear with nothing to clear, does not fill the stack with
        steps that do nothing.
        """
        if not description or self.tool_stack is None or before is None:
            return
        after = self.prompt_state()
        if after is None or after == before:
            return
        self.tool_stack.push(PromptStateAction(self, before, after, description=description))

    def _record_working_area(self, description="opening the work area"):
        """Start a fresh stack on the work area that was just opened.

        The previous session's steps describe a work area that no longer exists,
        so they go. Nothing is recorded when the work area failed to open, which
        is why this is called after the attempt rather than instead of it.
        """
        if self.tool_stack is None:
            return
        self.tool_stack.clear()
        if getattr(self, 'working_area', None) is not None:
            self.tool_stack.push(WorkAreaAction(self, description=description))

    def tool_undo(self):
        """Undo one of this tool's own steps.

        Returns:
            bool: True when a step was undone, so the caller leaves the
            annotation window's stack alone. False means this tool had nothing
            of its own pending and the 2D undo should run instead.
        """
        return self._run_tool_action(redo=False)

    def tool_redo(self):
        """Redo one of this tool's own steps. See tool_undo."""
        return self._run_tool_action(redo=True)

    def _run_tool_action(self, redo):
        """Step the tool's stack and say what moved."""
        stack = self.tool_stack
        if stack is None:
            return False
        if not (stack.can_redo() if redo else stack.can_undo()):
            return False

        action = stack.redo() if redo else stack.undo()
        if action is None:
            return False

        # Named, like the 2D undo: a prompt coming back somewhere off screen is
        # otherwise indistinguishable from the key having done nothing. The hint
        # returns on its own once this message expires.
        try:
            verb = "Redid" if redo else "Undid"
            self.main_window.status_bar.showMessage(f"{verb} {action.label}.", 4000)
        except Exception:
            pass
        return True

    def create_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Create and display a cursor annotation at the given position.
        Subclasses should override this method to create the appropriate annotation type.
        
        Args:
            scene_pos: Position in scene coordinates where to create the annotation
        """
        pass

    def create_cursor_preview_item(self, u: float, v: float):
        """
        Create a QGraphicsItem representing this tool's cursor at image pixel (u, v).
        Used by the cursor preview propagation system to display the cursor
        on consumer canvases at the projected position.

        Subclasses that participate in cursor preview propagation should override
        this method and return a fully styled, positioned QGraphicsItem.
        The item must NOT be added to a scene — BaseCanvas does that.

        Returns:
            QGraphicsItem, or None if this tool does not support cursor preview.
        """
        return None
        
    def update_cursor_annotation(self, scene_pos: QPointF = None):
        """
        Update the existing cursor annotation to a new position or with new properties.
        Subclasses should override this method to update the annotation appropriately.
        
        Args:
            scene_pos: New position for the cursor annotation
        """
        if self.cursor_annotation:
            self.clear_cursor_annotation()
            self.create_cursor_annotation(scene_pos)
    
    def clear_cursor_annotation(self):
        """
        Clear the current cursor annotation if it exists.
        """
        if self.cursor_annotation:
            self.cursor_annotation.delete()
            self.cursor_annotation = None
            
    def draw_crosshair(self, scene_pos):
        """
        Draw crosshair guides at the current cursor position.
        
        Args:
            scene_pos: Position in scene coordinates where to draw the crosshair
        """
        # Only draw if we have an active image and scene position
        if (
            not self.show_crosshair
            or not self.annotation_window.active_image
            or not scene_pos
        ):
            return

        # Remove any existing crosshair lines
        self.clear_crosshair()
        
        # Get image bounds
        image_rect = self.annotation_window.get_image_rect()
        
        # Create horizontal line across the full width of the image
        self.h_crosshair_line = QGraphicsLineItem(image_rect.left(), scene_pos.y(), image_rect.right(), scene_pos.y())
        pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.h_crosshair_line.setPen(pen)
        self.h_crosshair_line.setZValue(TOOL_OVERLAY_Z)
        self.annotation_window.scene.addItem(self.h_crosshair_line)
        
        # Create vertical line across the full height of the image
        self.v_crosshair_line = QGraphicsLineItem(scene_pos.x(), image_rect.top(), scene_pos.x(), image_rect.bottom())
        pen = QPen(QColor(255, 255, 255, 180), 1, Qt.DashLine)
        pen.setCosmetic(True)
        self.v_crosshair_line.setPen(pen)
        self.v_crosshair_line.setZValue(TOOL_OVERLAY_Z)
        self.annotation_window.scene.addItem(self.v_crosshair_line)
        
    def clear_crosshair(self):
        """Remove any crosshair guide lines from the scene."""
        if self.h_crosshair_line and self.h_crosshair_line.scene():
            self.annotation_window.scene.removeItem(self.h_crosshair_line)
            self.h_crosshair_line = None
        if self.v_crosshair_line and self.v_crosshair_line.scene():
            self.annotation_window.scene.removeItem(self.v_crosshair_line)
            self.v_crosshair_line = None

    def update_crosshair(self, scene_pos):
        """
        Update the crosshair position. This is a convenience method that
        clears and redraws the crosshair.
        
        Args:
            scene_pos: New position for the crosshair
        """
        self.clear_crosshair()
        self.draw_crosshair(scene_pos)