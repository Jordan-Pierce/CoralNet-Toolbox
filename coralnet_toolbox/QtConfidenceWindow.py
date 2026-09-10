import time
import warnings

from PyQt5.QtGui import QColor, QPainter, QCursor, QPainterPath, QPen, QFontMetrics
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer, QSize
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QBoxLayout, QSizePolicy, QLabel, QHBoxLayout, QFrame,
                             QPushButton, QMenu, QToolBar, QStatusBar)

from coralnet_toolbox.utilities import scale_pixmap

from coralnet_toolbox.MetaData.QtBuiltInFields import compute_builtin_fields
from coralnet_toolbox.MetaData.QtBuiltInFields import format_unconvertible_note

from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


BAR_TRACK_COLOR = app_theme.SURFACE_COLOR
ROW_HOVER_COLOR = app_theme.SURFACE_ELEVATED_COLOR
# Amber flags work that still needs a human: unverified annotations, close calls
ATTENTION_COLOR = QColor("#ffd479")
# A top-1 lead over top-2 smaller than this, in percentage points, is flagged as a close call
LOW_MARGIN_PTS = 10.0
# The crop sits beside the bars only when that still leaves the bars at least this wide
MIN_SIDE_BY_SIDE_BAR_WIDTH = 280

# The app stylesheet gives QPushButton no checked state, so on/off toggles here
# borrow the one QToolButton gets
TOGGLE_BUTTON_STYLE = (
    "QPushButton { padding: 0px; margin: 0px; } "
    f"QPushButton:checked {{ background-color: {app_theme.ACCENT_SOFT_COLOR.name()}; "
    f"border-color: {app_theme.ACCENT_COLOR.name()}; }}"
)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ConfidenceBar(QFrame):
    """A rounded bar: a dark track, filled in the class colour up to the confidence."""

    def __init__(self, label, confidence, emphasized=False, parent=None, animate=True):
        """Initialize the ConfidenceBar widget."""
        super().__init__(parent)

        self.label = label
        self.confidence = confidence
        self.color = label.color
        # The top prediction is drawn at full strength, the runners-up dimmer
        self.emphasized = emphasized
        self.setFixedHeight(app_theme.scale_int(12))
        # Clicks belong to the whole row, see ConfidenceRow
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._fill_width = 0
        self.target_fill_width = 0  # Will be set in resizeEvent

        # Cleared when this window is being rebuilt faster than the animation
        # could finish anyway -- see ConfidenceWindow.create_bar_chart.
        self.animate = animate

        # Animation will be created and started in the first resizeEvent
        self.animation = None

    def get_fill_width(self):
        """Getter for the fill_width property used by the animation."""
        return self._fill_width

    def set_fill_width(self, value):
        """Setter for the fill_width property used by the animation."""
        self._fill_width = value
        self.update()  # Trigger a repaint whenever the value changes

    # This property allows QPropertyAnimation to animate the fill width
    fill_width = pyqtProperty(int, fget=get_fill_width, fset=set_fill_width)

    def resizeEvent(self, event):
        """Handle resize to recalculate target fill width and start animation."""
        super().resizeEvent(event)
        # Calculate the target fill width based on the current widget width and confidence
        new_target = int(self.width() * (self.confidence / 100))

        # Qt delivers several resize events while a layout settles, and this
        # used to restart a 500 ms animation from zero on every one of them --
        # so a bar could be re-animating long after it had finished, and a dock
        # drag turned into a wall of restarts. Only an actual change in target
        # is worth re-animating.
        if new_target == self.target_fill_width and self.animation is not None:
            return

        self.target_fill_width = new_target

        # Stop any existing animation
        if self.animation is not None:
            self.animation.stop()

        # Start animation from current position to target
        self.start_animation()

    def start_animation(self):
        """Start the fill animation."""
        if not self.animate:
            # Snap straight to the answer. Half a second of easing is charming
            # once and a liability when the window is being rebuilt on every
            # keypress of a held-down cycle shortcut.
            if self.animation is not None:
                self.animation.stop()
            self._fill_width = self.target_fill_width
            self.update()
            return

        if self.target_fill_width <= 0:
            # Stop any existing animation
            if self.animation is not None:
                self.animation.stop()

            # Explicitly set the fill width to 0 and trigger a repaint
            self._fill_width = 0
            self.update()
            return

        self.animation = QPropertyAnimation(self, b"fill_width")
        self.animation.setDuration(500)  # 500ms duration
        self.animation.setStartValue(0)  # Note: This could be self._fill_width for a smoother resume
        self.animation.setEndValue(self.target_fill_width)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # Smooth easing
        self.animation.start()

    def paintEvent(self, event):
        """Handle the paint event to draw the confidence bar."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        if width < 1 or height < 1:
            return

        track = QRectF(0, 0, width, height)
        radius = height / 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(BAR_TRACK_COLOR)
        painter.drawRoundedRect(track, radius, radius)

        fill_width = min(self._fill_width, width)
        if fill_width <= 0:
            return

        # Clip the fill to the track's outline instead of rounding the fill
        # itself: a rounded fill can be no narrower than it is tall, which
        # would overstate every confidence under a few percent.
        outline = QPainterPath()
        outline.addRoundedRect(track, radius, radius)
        painter.setClipPath(outline)

        fill = QColor(self.color)
        fill.setAlpha(255 if self.emphasized else 170)
        painter.setBrush(fill)
        painter.drawRect(QRectF(0, 0, fill_width, height))


class ConfidenceRow(QWidget):
    """One top-k row -- rank, class, bar, percentage -- clickable as a whole.

    Only the bar used to take clicks, a thin strip beside a name and a number
    that looked every bit as clickable.
    """
    barClicked = pyqtSignal(object)  # Define a signal that takes an object (label)

    def __init__(self, confidence_window, label, rank_pixmap, confidence, emphasized=False,
                 animate=True, parent=None):
        """Initialize the ConfidenceRow widget."""
        super().__init__(parent)
        self.confidence_window = confidence_window
        self.label = label
        self._hovered = False

        self.setFixedHeight(app_theme.scale_int(24))
        self.setToolTip(label.long_label_code)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(app_theme.scale_int(4), 0, app_theme.scale_int(6), 0)
        layout.setSpacing(app_theme.scale_int(8))

        # Rank, which is also the number key that picks this class
        rank_label = QLabel()
        rank_label.setPixmap(rank_pixmap)
        rank_label.setFixedSize(app_theme.scale_int(14), app_theme.scale_int(14))

        # Class Label, fixed so every bar starts at the same x
        class_width = confidence_window.class_label_width
        class_label = QLabel()
        class_label.setFixedWidth(class_width)
        class_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        font = class_label.font()
        font.setBold(emphasized)
        class_label.setFont(font)
        # Elide with font metrics so the text always fits the fixed width
        class_label.setText(class_label.fontMetrics().elidedText(label.short_label_code, Qt.ElideRight, class_width))

        # Progress Bar
        self.bar = ConfidenceBar(label, confidence, emphasized=emphasized, animate=animate)

        # Percentage, fixed so every bar ends at the same x
        text_color = app_theme.TEXT_PRIMARY_COLOR if emphasized else app_theme.TEXT_SECONDARY_COLOR
        percentage_label = QLabel(f"{confidence:.1f}%")
        percentage_label.setFixedWidth(confidence_window.percentage_label_width)
        percentage_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        percentage_label.setStyleSheet(f"color: {text_color.name()};")

        for child in (rank_label, class_label, percentage_label):
            child.setAttribute(Qt.WA_TransparentForMouseEvents)

        layout.addWidget(rank_label)
        layout.addWidget(class_label)
        layout.addWidget(self.bar, 1)  # 1 allows the bar to stretch and absorb extra space
        layout.addWidget(percentage_label)

    def _selectable(self):
        """Bars only relabel while the Select tool is active."""
        return self.confidence_window.main_window.annotation_window.selected_tool == "select"

    def paintEvent(self, event):
        """Highlight the row under the cursor when a click would do something."""
        if not (self._hovered and self._selectable()):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(ROW_HOVER_COLOR)
        painter.drawRoundedRect(QRectF(self.rect()), 4, 4)

    def mousePressEvent(self, event):
        """Handle mouse press events on the row."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            # self.handle_click() # <-- DO NOT CALL DIRECTLY

            # Defer the click handling. This lets the mousePressEvent finish
            # before the widget is potentially deleted by the click's action.
            QTimer.singleShot(0, self.handle_click)

    def handle_click(self):
        """Handle the logic when the row is clicked."""
        # Check if the Selector tool is active
        if self._selectable():
            # Emit the signal with the label object
            self.barClicked.emit(self.label)
            # Set focus to the confidence window for keyboard events
            self.confidence_window.setFocus()

    def enterEvent(self, event):
        """Handle mouse enter events to change the cursor."""
        super().enterEvent(event)
        self._hovered = True
        # Change cursor based on the active tool
        if self._selectable():
            self.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.ForbiddenCursor))
        self.update()

    def leaveEvent(self, event):
        """Handle mouse leave events to reset the cursor."""
        super().leaveEvent(event)
        self._hovered = False
        self.setCursor(QCursor(Qt.ArrowCursor))  # Reset to the default cursor
        self.update()


class CropPreview(QWidget):
    """The selected annotation's crop, aspect-fit into a rounded frame.

    Replaces a QGraphicsView that only ever held one pixmap. The view brought
    a grey letterbox and scrollbars, and fitted a crop capped at 256 px back up
    to whatever size the dock happened to be.
    """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the CropPreview widget."""
        super().__init__(parent)
        self._pixmap = None
        self._scaled = None  # _pixmap smooth-scaled to the current frame, rebuilt on resize
        self._accent = None
        self._caption = ""
        self._placeholder = ""  # Shown in place of a crop
        self._top_aligned = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        """Prefer a comfortable square."""
        return QSize(app_theme.scale_int(160), app_theme.scale_int(160))

    def minimumSizeHint(self):
        """Allow the preview to shrink a long way before it forces the dock larger."""
        return QSize(app_theme.scale_int(48), app_theme.scale_int(48))

    def set_crop(self, pixmap, caption="", placeholder=""):
        """Show a crop, or the empty state -- with placeholder text, if given -- when pixmap is None."""
        self._pixmap = pixmap if pixmap is not None and not pixmap.isNull() else None
        self._scaled = None
        self._caption = caption if self._pixmap is not None else ""
        self._placeholder = placeholder
        self.setCursor(QCursor(Qt.PointingHandCursor if self._pixmap is not None else Qt.ArrowCursor))
        self.update()

    def set_accent(self, color):
        """Frame the crop in a class colour -- the top prediction's -- or neutrally when None."""
        self._accent = QColor(color) if color is not None else None
        self.update()

    def set_top_aligned(self, top_aligned):
        """Pin the crop to the top, to line up with bars beside it, rather than centring it."""
        if top_aligned != self._top_aligned:
            self._top_aligned = top_aligned
            self.update()

    def clear(self):
        """Return to the empty state."""
        self._accent = None
        self.setToolTip("")
        self.set_crop(None)

    def _image_rect(self):
        """The aspect-fit rect the crop is drawn into."""
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if self._pixmap is None or area.width() <= 0 or area.height() <= 0:
            return area
        scale = min(area.width() / self._pixmap.width(), area.height() / self._pixmap.height())
        width = self._pixmap.width() * scale
        height = self._pixmap.height() * scale
        left = area.center().x() - width / 2
        top = area.top() if self._top_aligned else area.center().y() - height / 2
        return QRectF(left, top, width, height)

    def paintEvent(self, event):
        """Paint the crop, its class-coloured frame and its size caption."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        radius = app_theme.scale_int(6)

        if self._pixmap is None:
            area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            if area.width() < 32 or area.height() < 32:
                return
            pen = QPen(app_theme.SURFACE_BORDER_COLOR)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(area, radius, radius)
            if self._placeholder:
                painter.setPen(app_theme.TEXT_MUTED_COLOR)
                painter.drawText(area, Qt.AlignCenter, self._placeholder)
            return

        target = self._image_rect()
        if target.width() < 8 or target.height() < 8:
            return

        # Scale once per size rather than on every paint, at device resolution
        # so the crop stays sharp on high-DPI screens
        ratio = self.devicePixelRatioF()
        size = QSize(max(1, round(target.width() * ratio)), max(1, round(target.height() * ratio)))
        if self._scaled is None or self._scaled.size() != size:
            self._scaled = self._pixmap.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        outline = QPainterPath()
        outline.addRoundedRect(target, radius, radius)
        painter.save()
        painter.setClipPath(outline)
        painter.fillRect(target, app_theme.BACKGROUND_COLOR)  # Behind any transparent parts of the crop
        painter.drawPixmap(target, self._scaled, QRectF(self._scaled.rect()))
        painter.restore()

        # Frame in the top prediction's colour, as the old view's border was
        if self._accent is not None:
            pen = QPen(self._accent)
            pen.setWidthF(2.0)
        else:
            pen = QPen(app_theme.SURFACE_BORDER_COLOR)
            pen.setWidthF(1.0)
        inset = pen.widthF() / 2
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(target.adjusted(inset, inset, -inset, -inset), radius, radius)

        self._paint_caption(painter, target)

    def _paint_caption(self, painter, target):
        """Paint the crop size in a small pill in the bottom-right corner, if it fits."""
        if not self._caption:
            return
        font = painter.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(6.0, font.pointSizeF() - 1.5))
        painter.setFont(font)
        metrics = QFontMetrics(font)
        pad_x, pad_y, inset = app_theme.scale_int(5), app_theme.scale_int(1), app_theme.scale_int(5)
        pill_width = metrics.horizontalAdvance(self._caption) + 2 * pad_x
        pill_height = metrics.height() + 2 * pad_y
        if pill_width + 2 * inset > target.width() or pill_height + 2 * inset > target.height() / 2:
            return
        pill = QRectF(target.right() - inset - pill_width, target.bottom() - inset - pill_height,
                      pill_width, pill_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(pill, pill_height / 2, pill_height / 2)
        painter.setPen(app_theme.TEXT_PRIMARY_COLOR)
        painter.drawText(pill, Qt.AlignCenter, self._caption)

    def mousePressEvent(self, event):
        """Emit clicked for a left-click on a crop."""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self.clicked.emit()


class ConfidenceWindow(QWidget):
    # Rebuilds closer together than this skip the bar fill animation; see
    # create_bar_chart.
    ANIMATION_DEBOUNCE_S = 0.5

    def __init__(self, main_window, parent=None):
        """Initialize the ConfidenceWindow widget."""
        super().__init__(parent)
        self.main_window = main_window
        self.label_window = main_window.label_window

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Replaced QGroupBox with a QFrame to eliminate awkward top padding
        self.container = QFrame(self)
        self.container.setFrameShape(QFrame.StyledPanel)

        # Main layout for the container
        self.containerLayout = QVBoxLayout(self.container)
        self.containerLayout.setContentsMargins(6, 6, 6, 6) # Give it some breathing room
        self.containerLayout.setSpacing(6)

        # Crops are capped at this size once, then smooth-scaled to the preview
        self.max_graphic_size = 512
        self.crop_preview_visible = True

        self.bar_chart_widget = None
        self.bar_chart_layout = None

        # Prepare variables
        self.annotation = None
        self.user_confidence = None
        self.machine_confidence = None
        self.chart_dict = None
        self.confidence_bar_labels = []
        self._last_chart_build = 0.0

        # Get and store the icons
        self.user_icon = get_icon("user.svg")
        self.machine_icon = get_icon("machine.svg")
        self.prev_icon = get_icon("left.svg")
        self.next_icon = get_icon("right.svg")
        self.image_icon = get_icon("image.svg")

        self.top_k_icons = {
            "1": get_icon("1.svg").pixmap(app_theme.scale_size(12)),
            "2": get_icon("2.svg").pixmap(app_theme.scale_size(12)),
            "3": get_icon("3.svg").pixmap(app_theme.scale_size(12)),
            "4": get_icon("4.svg").pixmap(app_theme.scale_size(12)),
            "5": get_icon("5.svg").pixmap(app_theme.scale_size(12))
        }

        self.icon_button_size = app_theme.scale_int(26)
        self.icon_button_icon_size = 16

        # Fixed widths so all bars start and end at the same x
        self.class_label_width = app_theme.scale_int(90)
        self.percentage_label_width = app_theme.scale_int(56)

        # 1. Header: navigation, what is on display, and the view toggles
        self.prev_button = QPushButton(self.prev_icon, "")
        self.prev_button.setToolTip("Select an annotation to enable navigation")
        self.prev_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.prev_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.prev_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.prev_button.clicked.connect(self.on_prev_clicked)

        self.next_button = QPushButton(self.next_icon, "")
        self.next_button.setToolTip("Select an annotation to enable navigation")
        self.next_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.next_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.next_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.next_button.clicked.connect(self.on_next_clicked)

        # Annotation type and review state. Ignored horizontally so a long
        # string can never hold the dock open.
        self.title_label = QLabel(self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setTextFormat(Qt.RichText)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.toggle_button = QPushButton(self)
        self.toggle_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.toggle_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.toggle_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.toggle_state = False  # False = user, True = machine
        self.toggle_button.setIcon(self.user_icon)
        self.toggle_button.clicked.connect(self.toggle_user_machine_confidence_icon)
        self.set_user_icon(False)

        self.crop_toggle_button = QPushButton(self.image_icon, "")
        self.crop_toggle_button.setCheckable(True)
        self.crop_toggle_button.setChecked(self.crop_preview_visible)
        self.crop_toggle_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.crop_toggle_button.setStyleSheet(TOGGLE_BUTTON_STYLE)
        self.crop_toggle_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.crop_toggle_button.toggled.connect(self.set_crop_preview_visible)

        # Pack controls into a horizontal row
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.prev_button)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.crop_toggle_button)
        header_layout.addWidget(self.next_button)

        self.containerLayout.addLayout(header_layout)

        # 2. Body: the crop and the bars, side by side when there is room for
        # both (see _update_body_direction), stacked otherwise
        self.crop_preview = CropPreview(self)
        self.crop_preview.clicked.connect(self.on_crop_clicked)
        self.init_bar_chart_widget()

        self.body_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(8)
        self.body_layout.addWidget(self.crop_preview, 1)
        self.body_layout.addWidget(self.bar_chart_widget, 0)
        self.containerLayout.addLayout(self.body_layout, 1)

        # Add the main container to the window's layout
        self.layout.addWidget(self.container)

        # Start in the empty state
        self.clear_display()
        self.set_crop_preview_visible(self.crop_preview_visible)

    # --- DOCK WRAPPER HOOKS ---

    def create_menu(self) -> QMenu:
        """Create a contextual menu specific to this Window."""
        # Example for the future:
        # menu = QMenu("Options", self)
        # menu.addAction("Export Data...")
        # return menu
        return None

    def create_top_toolbar(self) -> QToolBar:
        """Create a top toolbar specific to this Window."""
        # Example for the future:
        # toolbar = QToolBar("Tools")
        # toolbar.addAction("Toggle Details")
        # return toolbar
        return None

    def create_bottom_status_bar(self) -> QStatusBar:
        """Create a status bar specific to this Window."""
        # Example for the future:
        # status = QStatusBar()
        # status.showMessage("Ready")
        # return status
        return None

    def resizeEvent(self, event):
        """Handle resize events for the widget."""
        super().resizeEvent(event)
        self._update_body_direction()

    def keyPressEvent(self, event):
        """Handle key press events for 1-5 to select a confidence bar."""
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_5:
            idx = (key - Qt.Key_1)  # 0-based index
            if hasattr(self, "confidence_bar_labels") and idx < len(self.confidence_bar_labels):
                label = self.confidence_bar_labels[idx]
                self.handle_bar_click(label)
        else:
            super().keyPressEvent(event)

    def _update_body_direction(self):
        """Put the crop beside the bars when that leaves the bars readable, above them otherwise.

        The bars carry fixed-width name and percentage columns, so a crop beside
        them in a narrow dock squeezes the bar itself down to a sliver. Beside
        is only worth it once the dock is wider than it is tall by at least a
        readable bar column.
        """
        wide = self.width() >= self.height() + app_theme.scale_int(MIN_SIDE_BY_SIDE_BAR_WIDTH)
        direction = QBoxLayout.LeftToRight if wide else QBoxLayout.TopToBottom
        if self.body_layout.direction() == direction:
            return
        self.body_layout.setDirection(direction)
        self.body_layout.setStretchFactor(self.crop_preview, 2 if wide else 1)
        self.body_layout.setStretchFactor(self.bar_chart_widget, 3 if wide else 0)
        self.crop_preview.set_top_aligned(wide)

    def set_navigation_enabled(self, enabled):
        """Enable or disable annotation navigation buttons."""
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

        if not enabled:
            self.prev_button.setToolTip("Select an annotation to enable navigation")
            self.next_button.setToolTip("Select an annotation to enable navigation")
        else:
            self.prev_button.setToolTip("Previous Annotation")
            self.next_button.setToolTip("Next Annotation")

    def on_prev_clicked(self):
        """Handle previous button click."""
        self.main_window.annotation_window.cycle_annotations(-1)
        self.setFocus()

    def on_next_clicked(self):
        """Handle next button click."""
        self.main_window.annotation_window.cycle_annotations(1)
        self.setFocus()

    def on_crop_clicked(self):
        """Centre the annotation window on the annotation on display."""
        annotation = self.annotation
        annotation_window = self.main_window.annotation_window
        # Only for the image being shown: centring builds the annotation's
        # graphics item in the current scene if it has none
        if annotation is None or annotation.image_path != annotation_window.current_image_path:
            return
        annotation_window.center_on_annotation(annotation)

    def init_bar_chart_widget(self):
        """Initialize the widget and layout for the confidence bar chart."""
        self.bar_chart_widget = QWidget()
        column_layout = QVBoxLayout(self.bar_chart_widget)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(4)

        self.bar_chart_layout = QVBoxLayout()
        self.bar_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_chart_layout.setSpacing(2)
        column_layout.addLayout(self.bar_chart_layout)

        # How far the top class leads the runner-up
        self.margin_label = QLabel()
        self.margin_label.setTextFormat(Qt.RichText)
        self.margin_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.margin_label.setContentsMargins(0, 0, app_theme.scale_int(6), 0)
        column_layout.addWidget(self.margin_label)

        # Keep the rows at the top however much height the column is given
        column_layout.addStretch(1)

    def set_crop_preview_visible(self, visible):
        """Show or hide the crop, leaving the bars the whole dock when it is hidden."""
        self.crop_preview_visible = bool(visible)
        if self.crop_toggle_button.isChecked() != self.crop_preview_visible:
            self.crop_toggle_button.setChecked(self.crop_preview_visible)
        self.crop_preview.setVisible(self.crop_preview_visible)
        self.crop_toggle_button.setToolTip("Hide Crop Preview" if self.crop_preview_visible else "Show Crop Preview")

        if self.crop_preview_visible:
            # Crops are not rendered while the preview is hidden, so the one on
            # display has to be caught up now
            self._load_crop(self.annotation)
        else:
            self.crop_preview.set_crop(None)

    def toggle_user_machine_confidence_icon(self):
        """Toggle the button icon and switch between user/machine confidences."""
        if not (self.user_confidence and self.machine_confidence):
            return  # Nothing to toggle

        self.toggle_state = not self.toggle_state
        if self.toggle_state:
            self.chart_dict = self.machine_confidence
            self.set_machine_icon(enabled=True)
        else:
            self.chart_dict = self.user_confidence
            self.set_user_icon(enabled=True)
        self.create_bar_chart()

    def set_user_icon(self, enabled=True):
        """Set the button icon to user mode."""
        self.toggle_button.setIcon(self.user_icon)
        self.toggle_button.setToolTip("Viewing User Confidences")
        self.toggle_button.setEnabled(enabled)
        self.toggle_state = False

    def set_machine_icon(self, enabled=True):
        """Set the button icon to machine mode."""
        self.toggle_button.setIcon(self.machine_icon)
        self.toggle_button.setToolTip("Viewing Machine Confidences")
        self.toggle_button.setEnabled(enabled)
        self.toggle_state = True

    def refresh_scaling(self):
        """Refresh icon and control sizes after a UI scale change."""
        self.icon_button_size = app_theme.scale_int(26)
        self.icon_button_icon_size = 16

        self.class_label_width = app_theme.scale_int(90)
        self.percentage_label_width = app_theme.scale_int(56)

        self.prev_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.prev_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.prev_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.next_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.next_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.next_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.toggle_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.toggle_button.setStyleSheet("padding: 0px; margin: 0px;")
        self.toggle_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.crop_toggle_button.setFixedSize(self.icon_button_size, self.icon_button_size)
        self.crop_toggle_button.setStyleSheet(TOGGLE_BUTTON_STYLE)
        self.crop_toggle_button.setIconSize(app_theme.scale_size(self.icon_button_icon_size))
        self.margin_label.setContentsMargins(0, 0, app_theme.scale_int(6), 0)

        self.top_k_icons = {
            "1": get_icon("1.svg").pixmap(app_theme.scale_size(12)),
            "2": get_icon("2.svg").pixmap(app_theme.scale_size(12)),
            "3": get_icon("3.svg").pixmap(app_theme.scale_size(12)),
            "4": get_icon("4.svg").pixmap(app_theme.scale_size(12)),
            "5": get_icon("5.svg").pixmap(app_theme.scale_size(12))
        }

        if self.toggle_state:
            self.set_machine_icon(self.toggle_button.isEnabled())
        else:
            self.set_user_icon(self.toggle_button.isEnabled())

        # Rows are built at the old scale; rebuild them at the new one
        if self.annotation:
            self.create_bar_chart()
        self._update_body_direction()

    def update_annotation(self, annotation):
        """Update the currently displayed annotation data."""
        if annotation:
            self.annotation = annotation
            self.user_confidence = annotation.user_confidence
            self.machine_confidence = annotation.machine_confidence

            # Annotation is verified and contains machine confidences
            if annotation.verified and self.machine_confidence:
                self.chart_dict = self.user_confidence
                self.set_user_icon(annotation.verified)         # enabled user icon

            # Annotation is not verified and contains machine confidences
            elif not annotation.verified and self.machine_confidence:
                self.chart_dict = self.machine_confidence
                self.set_machine_icon(annotation.verified)      # disabled machine icon

            # Annotation is verified and does not contain machine confidences
            elif annotation.verified and not self.machine_confidence:
                self.chart_dict = self.user_confidence
                self.set_user_icon(not annotation.verified)     # disabled user icon

            # Annotation is not verified and has no machine confidences either
            else:
                self.chart_dict = self.user_confidence
                self.set_user_icon(False)                       # disabled user icon

        else:
            self.set_user_icon(False)  # Disable user icon if no annotation is provided

    def refresh_display(self):
        """Refresh the confidence window display for the current annotation."""
        if self.annotation:
            # Update annotation data
            self.update_annotation(self.annotation)

            # Verification can have changed along with the label
            self._update_title(self.annotation)

            # Recreate the bar chart with updated data (and the crop's frame colour)
            self.create_bar_chart()

            # Recreate the tooltip to reflect any data changes (like scale)
            self.create_annotation_tooltip(self.annotation)

    def on_annotation_updated(self, updated_annotation):
        """Handle annotation update signal - refresh display if it's the currently shown annotation."""
        if self.annotation and updated_annotation.id == self.annotation.id:
            self.refresh_display()

    def display_cropped_image(self, annotation):
        """Display an annotation: its type and state, its crop, and its confidences."""
        try:
            self.clear_display()
            if annotation is None:
                return

            self.update_annotation(annotation)
            self._update_title(annotation)

            # Create tooltip with annotation information
            self.create_annotation_tooltip(annotation)

            # Create the bar charts
            self.create_bar_chart()

            # Skipped while the preview is hidden -- building the crop graphic is
            # most of what displaying an annotation used to cost
            if self.crop_preview_visible:
                self._load_crop(annotation)

            # Enable navigation buttons
            self.set_navigation_enabled(True)

        except Exception as e:
            # Cropped image is None or some other error occurred
            print(f"Error displaying cropped image: {e}")
            # Ensure buttons are disabled if loading fails
            self.set_navigation_enabled(False)

    def _load_crop(self, annotation):
        """Render an annotation's crop into the preview."""
        if annotation is None:
            self.crop_preview.set_crop(None)
            return

        graphic = annotation.get_cropped_image_graphic() if annotation.cropped_image else None
        if graphic is None or graphic.isNull():
            self.crop_preview.set_crop(None, placeholder="No crop available")
            return

        # Height x width, as everywhere else in the app
        crop = annotation.cropped_image
        caption = f"{crop.height()} × {crop.width()}"
        self.crop_preview.set_crop(scale_pixmap(graphic, self.max_graphic_size), caption)

    def _update_title(self, annotation):
        """Say what is on display: the annotation's type and whether it has been reviewed."""
        if annotation is None:
            self.title_label.setText(
                f'<span style="color:{app_theme.TEXT_SECONDARY_COLOR.name()}">No annotation selected</span>'
            )
            return

        kind = type(annotation).__name__.replace("Annotation", "") or "Annotation"
        if annotation.verified:
            state = f'<span style="color:{app_theme.TEXT_SECONDARY_COLOR.name()}">verified</span>'
        else:
            state = f'<span style="color:{ATTENTION_COLOR.name()}">unverified</span>'
        self.title_label.setText(
            f'<span style="color:{app_theme.TEXT_PRIMARY_COLOR.name()}">{kind}</span> · {state}'
        )

    def create_annotation_tooltip(self, annotation):
        """Create a formatted tooltip for the annotation, on the crop and on the header."""
        # Computed by the shared helper so this tooltip and the Metadata dock
        # can never disagree about a derived value.
        fields, unconvertible_units = compute_builtin_fields(annotation, self.main_window)

        tooltip_parts = [f"<b>{name}:</b> {value}" for name, value in fields.items()]

        note = format_unconvertible_note(unconvertible_units)
        if note:
            tooltip_parts.append(f"<i>{note}</i>")

        # Set the tooltip. The header carries it too, so hiding the crop does
        # not hide the details with it.
        tooltip_text = "<br>".join(tooltip_parts)
        self.title_label.setToolTip(tooltip_text)
        self.crop_preview.setToolTip(tooltip_text + "<br><i>Click to center it in the Annotation window</i>")

    def create_bar_chart(self):
        """Create and populate the confidence bar chart."""
        # AnnotationWindow.cycle_annotations already skips its view animation
        # when the user is stepping through annotations faster than
        # ANIMATION_DEBOUNCE_S; this window never got the memo, so holding
        # Ctrl+Left/Right left five bars permanently mid-animation. Same rule,
        # applied here.
        now = time.monotonic()
        animate = (now - self._last_chart_build) >= self.ANIMATION_DEBOUNCE_S
        self._last_chart_build = now

        self.clear_layout(self.bar_chart_layout)
        self.confidence_bar_labels = []
        self._update_margin([])
        self.crop_preview.set_accent(None)

        if not self.chart_dict:
            return

        labels, confidences = self.get_chart_data()
        if not confidences:
            return

        # The top prediction sets the crop's frame colour and is drawn strongest
        top_index = confidences.index(max(confidences))
        self.crop_preview.set_accent(labels[top_index].color)

        for idx, (label, confidence) in enumerate(zip(labels, confidences)):
            self.add_bar_to_layout(label, confidence, idx + 1, emphasized=(idx == top_index), animate=animate)
            self.confidence_bar_labels.append(label)

        self._update_margin(confidences)

    def get_chart_data(self):
        """Retrieve the top 5 labels and confidences from the current chart dictionary."""
        keys = list(self.chart_dict.keys())[:5]
        return (
            keys,
            [conf_value * 100 for conf_value in self.chart_dict.values()][:5]
        )

    def add_bar_to_layout(self, label, confidence, top_k, emphasized=False, animate=True):
        """Create and add a row for one confidence bar to the layout."""
        row = ConfidenceRow(self, label, self.top_k_icons[str(top_k)], confidence,
                            emphasized=emphasized, animate=animate)
        row.barClicked.connect(self.handle_bar_click)
        self.bar_chart_layout.addWidget(row)

    def _update_margin(self, confidences):
        """Show how far the top class leads the runner-up; a narrow lead is worth a second look."""
        if len(confidences) < 2:
            self.margin_label.setText("")
            self.margin_label.setToolTip("")
            return

        first, second = sorted(confidences, reverse=True)[:2]
        margin = first - second
        if margin < LOW_MARGIN_PTS:
            text, color = "Close call", ATTENTION_COLOR
        else:
            text, color = "Margin", app_theme.TEXT_SECONDARY_COLOR
        self.margin_label.setText(f'<span style="color:{color.name()}">{text}: {margin:.1f} pts</span>')
        self.margin_label.setToolTip("How far the top class leads the second, in percentage points.\n"
                                     f"Under {LOW_MARGIN_PTS:.0f} points is flagged as a close call.")

    def handle_bar_click(self, label):
        """Handle clicks on a confidence bar to update the annotation."""
        # Guard clause: If no annotation is selected, do nothing.
        if not self.annotation:
            return

        # Store a local reference to the annotation.
        # This is crucial because unselect_annotation() will call clear_display()
        # and set self.annotation to None.
        annotation_to_update = self.annotation

        # Update the confidences to whichever bar was selected
        annotation_to_update.update_user_confidence(label)
        # Update the label to whichever bar was selected
        annotation_to_update.update_label(label)

        # Notify all subscribers (including the EmbeddingViewer) that this
        # annotation's label has changed.  The annotationLabelChanged signal is
        # the canonical way viewers such as the EmbeddingViewer learn about
        # label updates; without it, _refresh_point_colors is never called and
        # the scatter-plot dot colour stays stale until the next full refresh.
        try:
            self.main_window.annotation_window.annotationLabelChanged.emit(
                annotation_to_update.id,
                label.id if hasattr(label, 'id') else str(label)
            )
        except Exception:
            pass

        # Update the search bars
        self.main_window.image_window.update_search_bars()

        # Refresh in place rather than unselecting and reselecting.
        #
        # That round trip existed only to force a redraw, and it was expensive
        # out of all proportion to a label change: unselect_annotation tears
        # down the annotation's Qt items, clears this window (dropping
        # self.annotation), and rebuilds its colour group in the phantom layer;
        # select_annotation then rebuilds all of it, re-crops the image, and
        # rebuilds this chart. The visible result was a flicker on every bar
        # click. The annotation never actually stopped being selected, so say so.
        annotation_window = self.main_window.annotation_window
        annotation_window.selected_label = label
        # Keeps the Label Window's highlight in step, which is the one thing
        # reselecting used to be doing that a redraw does not.
        annotation_window.labelSelected.emit(label.id)
        # The annotation's colour group key changed with its label, so the
        # phantom layer needs the same patch a deselect would have given it.
        try:
            annotation_window.refresh_phantom_annotations(only_annotation=annotation_to_update)
        except Exception:
            pass
        annotation_window.viewport().update()

        self.refresh_display()

    def clear_layout(self, layout):
        """Remove all widgets from the specified layout."""
        for i in reversed(range(layout.count())):
            # Spacers and stretches have no widget; takeAt drops them all the same
            widget = layout.takeAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def clear_display(self):
        """
        Clears the crop, the bar chart and the header.
        """
        # Clear the crop
        self.crop_preview.clear()
        # Clear the bar chart layout
        self.clear_layout(self.bar_chart_layout)
        self.confidence_bar_labels = []
        self._update_margin([])
        # Back to the empty header, tooltip included
        self._update_title(None)
        self.title_label.setToolTip("")
        # Set the toggle button to user mode
        self.set_user_icon(False)
        # Disable navigation buttons
        self.set_navigation_enabled(False)
        # Clear the annotation reference, and the confidences that came with it
        self.annotation = None
        self.chart_dict = None
