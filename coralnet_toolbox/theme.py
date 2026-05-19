from __future__ import annotations

import math
import re
from pathlib import Path

# ── Must be set BEFORE QApplication ──────────────────────────────────────── #
from PyQt5.QtGui import QSurfaceFormat


def _configure_surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    fmt.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(fmt)


_configure_surface_format()
# ──────────────────────────────────────────────────────────────────────────── #

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication


# ─────────────────────────────────────────────────────────────────────────── #
#  Theme tokens                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

BACKGROUND_COLOR = QColor("#0f1117")
BACKGROUND_ALT_COLOR = QColor("#13161f")
SURFACE_COLOR = QColor("#1e2130")
SURFACE_ELEVATED_COLOR = QColor("#252a3d")
SURFACE_BORDER_COLOR = QColor("#2a2d3a")
TEXT_PRIMARY_COLOR = QColor("#d4d8e8")
TEXT_SECONDARY_COLOR = QColor("#8892b0")
TEXT_MUTED_COLOR = QColor("#556080")
TEXT_BRIGHT_COLOR = QColor("#ffffff")
ACCENT_COLOR = QColor("#3d7aed")
ACCENT_HOVER_COLOR = QColor("#5090ff")
ACCENT_SOFT_COLOR = QColor("#2347a0")
ACCENT_ALT_COLOR = QColor("#2e4a90")
DISABLED_COLOR = QColor("#555870")
SHADOW_COLOR = QColor(0, 0, 0, 80)

FONT_STACK = '"JetBrains Mono", "Cascadia Code", "Fira Code", monospace'
APP_FONT_FAMILY = "JetBrains Mono"
APP_FONT_SIZE = 10
ICON_DEFAULT_COLOR = TEXT_PRIMARY_COLOR
ICON_DIR = Path(__file__).resolve().parent / "Icons"
SPINBOX_UP_ARROW_ICON = (ICON_DIR / "spinbox-arrow-up.svg").as_posix()
SPINBOX_DOWN_ARROW_ICON = (ICON_DIR / "spinbox-arrow-down.svg").as_posix()

# Dock-window title-bar icons (used by the ADS docking system). Centralised so
# all four buttons read the same way — a clear ×, a hollow square for maximize,
# overlapping squares for restore, and an "out of box" arrow for detach/float.
DOCK_CLOSE_ICON = (ICON_DIR / "dock-close.svg").as_posix()
DOCK_MAXIMIZE_ICON = (ICON_DIR / "dock-maximize.svg").as_posix()
DOCK_RESTORE_ICON = (ICON_DIR / "dock-restore.svg").as_posix()
DOCK_DETACH_ICON = (ICON_DIR / "dock-detach.svg").as_posix()

SCALE_MODE_AUTO = "auto"
SCALE_MODE_MANUAL = "manual"
SCALE_PRESETS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SCALE_MIN = 0.75
SCALE_MAX = 2.0
SCALE_STEP = 0.25

_scale_mode = SCALE_MODE_MANUAL
_scale_factor = 1.0

_SCALEABLE_QSS_PROPERTIES = {
    "font-size",
    "padding",
    "padding-left",
    "padding-right",
    "padding-top",
    "padding-bottom",
    "margin",
    "margin-left",
    "margin-right",
    "margin-top",
    "margin-bottom",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "width",
    "height",
    "spacing",
    "border-radius",
    "icon-size",
    "letter-spacing",
    "text-indent",
}

_QSS_DECLARATION_PATTERN = re.compile(r"^(\s*)([A-Za-z-]+)\s*:\s*(.+);\s*$")
_QSS_LENGTH_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)px")


def _clamp_scale_factor(scale_factor: float) -> float:
    return max(SCALE_MIN, min(SCALE_MAX, float(scale_factor)))


def _format_scaled_px(value: float) -> str:
    return f"{max(1, int(math.floor(value + 0.5)))}px"


def get_scale_mode() -> str:
    return _scale_mode


def get_scale_factor() -> float:
    return _scale_factor


def set_scale_mode(scale_mode: str) -> None:
    if scale_mode not in (SCALE_MODE_AUTO, SCALE_MODE_MANUAL):
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    global _scale_mode
    _scale_mode = scale_mode


def set_scale_factor(scale_factor: float) -> float:
    global _scale_factor, _scale_mode
    _scale_factor = _clamp_scale_factor(scale_factor)
    _scale_mode = SCALE_MODE_MANUAL
    return _scale_factor


def scale_int(value: float) -> int:
    return max(1, int(math.floor(value * get_scale_factor() + 0.5)))


def scale_size(width: float, height: float | None = None) -> QSize:
    scaled_width = scale_int(width)
    scaled_height = scale_int(width if height is None else height)
    return QSize(scaled_width, scaled_height)


def scale_font(point_size: int | None = None, family: str | None = None, bold: bool | None = None) -> QFont:
    scaled_point_size = scale_int(point_size if point_size is not None else APP_FONT_SIZE)
    font = QFont(family or APP_FONT_FAMILY, scaled_point_size)
    if bold is not None:
        font.setBold(bold)
    return font


def resolve_auto_scale_factor(app: QApplication | None = None, widget=None) -> float:
    screen = None

    if widget is not None:
        try:
            window_handle = widget.windowHandle()
            if window_handle is not None:
                screen = window_handle.screen()
        except Exception:
            screen = None

    if screen is None and app is not None:
        try:
            screen = app.primaryScreen()
        except Exception:
            screen = None

        if screen is None:
            try:
                screens = app.screens()
                screen = screens[0] if screens else None
            except Exception:
                screen = None

    if screen is None:
        return 1.0

    try:
        dpi_scale = screen.logicalDotsPerInch() / 96.0
    except Exception:
        dpi_scale = 1.0

    try:
        geometry = screen.availableGeometry()
        width_scale = geometry.width() / 1920.0
        height_scale = geometry.height() / 1080.0
        resolution_scale = min(width_scale, height_scale)
    except Exception:
        resolution_scale = 1.0

    raw_scale = max(1.0, dpi_scale, resolution_scale)
    snapped_scale = math.ceil(raw_scale / SCALE_STEP) * SCALE_STEP
    return _clamp_scale_factor(snapped_scale)


def scale_qss(qss: str, scale_factor: float | None = None) -> str:
    factor = get_scale_factor() if scale_factor is None else _clamp_scale_factor(scale_factor)
    if factor == 1.0:
        return qss

    scaled_lines: list[str] = []
    for line in qss.splitlines():
        match = _QSS_DECLARATION_PATTERN.match(line)
        if not match:
            scaled_lines.append(line)
            continue

        indent, property_name, value = match.groups()
        if property_name.lower() not in _SCALEABLE_QSS_PROPERTIES:
            scaled_lines.append(line)
            continue

        scaled_value = _QSS_LENGTH_PATTERN.sub(
            lambda value_match: _format_scaled_px(float(value_match.group(1)) * factor),
            value,
        )
        scaled_lines.append(f"{indent}{property_name}: {scaled_value};")

    return "\n".join(scaled_lines)


# ─────────────────────────────────────────────────────────────────────────── #
#  Dark palette                                                               #
# ─────────────────────────────────────────────────────────────────────────── #


def build_palette() -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.Window, BACKGROUND_COLOR)
    pal.setColor(QPalette.WindowText, TEXT_PRIMARY_COLOR)
    pal.setColor(QPalette.Base, BACKGROUND_COLOR)
    pal.setColor(QPalette.AlternateBase, BACKGROUND_ALT_COLOR)
    pal.setColor(QPalette.ToolTipBase, SURFACE_COLOR)
    pal.setColor(QPalette.ToolTipText, TEXT_PRIMARY_COLOR)
    pal.setColor(QPalette.Text, TEXT_PRIMARY_COLOR)
    pal.setColor(QPalette.Button, SURFACE_COLOR)
    pal.setColor(QPalette.ButtonText, TEXT_PRIMARY_COLOR)
    pal.setColor(QPalette.BrightText, TEXT_BRIGHT_COLOR)
    pal.setColor(QPalette.Link, ACCENT_COLOR)
    pal.setColor(QPalette.Highlight, ACCENT_COLOR)
    pal.setColor(QPalette.HighlightedText, TEXT_BRIGHT_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.Window, BACKGROUND_ALT_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.Base, BACKGROUND_ALT_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.Button, SURFACE_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.WindowText, DISABLED_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.Text, DISABLED_COLOR)
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, DISABLED_COLOR)
    return pal


_build_dark_palette = build_palette


# ─────────────────────────────────────────────────────────────────────────── #
#  QSS stylesheet                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

STYLESHEET = """
/* ── Root ─────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #13161f;
    color: #d4d8e8;
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
}

/* ── Dock ──────────────────────────────────────────────────── */
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background: #1e2130;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7b8ec8;
    border-bottom: 1px solid #2a2d3a;
}

/* ── GroupBox ───────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 6px;
    background: #0f1117;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    top: 2px;
    color: #5a7ec8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Buttons ────────────────────────────────────────────────── */
QPushButton {
    background-color: #1e2130;
    color: #c8d0ea;
    border: 1px solid #2e3348;
    border-radius: 5px;
    padding: 5px 12px;
    font-size: 12px;
    min-height: 26px;
}
QPushButton:hover {
    background-color: #252a3d;
    border-color: #3d5aad;
    color: #e8ecff;
}
QPushButton:pressed {
    background-color: #1a1f30;
    border-color: #3d7aed;
}
QPushButton:disabled,
QToolButton:disabled {
    background-color: #151923;
    color: #6b738b;
    border-color: #252b3c;
}

QPushButton#primaryBtn {
    background-color: #1e3a6e;
    border-color: #3d7aed;
    color: #e0eaff;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #2347a0;
    border-color: #5090ff;
}

QPushButton#resetBtn {
    background-color: #1e2130;
    border: 1px solid #2e3348;
    border-radius: 4px;
    color: #7b8ec8;
    padding: 2px 4px;
    font-size: 13px;
    min-height: 20px;
}
QPushButton#resetBtn:hover {
    color: #aac0ff;
    border-color: #3d5aad;
}

/* ── Sliders ────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #252a3d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3d7aed;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #5090ff;
}
QSlider::sub-page:horizontal {
    background: #2e4a90;
    border-radius: 2px;
}

/* ── ComboBox ───────────────────────────────────────────────── */
QComboBox {
    background-color: #1e2130;
    border: 1px solid #2e3348;
    border-radius: 5px;
    padding: 4px 8px;
    color: #c8d0ea;
    min-height: 24px;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #7b8ec8;
    width: 0;
    height: 0;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #1a1d28;
    border: 1px solid #3d5aad;
    selection-background-color: #2347a0;
    color: #d4d8e8;
    outline: none;
}
QComboBox:disabled,
QLineEdit:disabled,
QTextEdit:disabled,
QPlainTextEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background-color: #151923;
    color: #6b738b;
    border: 1px solid #252b3c;
}
QComboBox:disabled::drop-down,
QSpinBox:disabled::up-button,
QSpinBox:disabled::down-button,
QDoubleSpinBox:disabled::up-button,
QDoubleSpinBox:disabled::down-button {
    background-color: #11151e;
    border-color: #252b3c;
}

/* ── CheckBox ───────────────────────────────────────────────── */
QCheckBox {
    spacing: 8px;
    color: #b0bada;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3a3f58;
    border-radius: 3px;
    background: #1a1d28;
}
QCheckBox::indicator:checked {
    background-color: #3d7aed;
    border-color: #3d7aed;
}
QCheckBox::indicator:hover {
    border-color: #5090ff;
}
QCheckBox:disabled,
QRadioButton:disabled {
    color: #6b738b;
}
QCheckBox:disabled::indicator,
QRadioButton:disabled::indicator {
    background: #11151e;
    border-color: #252b3c;
}

/* ── Labels ─────────────────────────────────────────────────── */
QLabel#monoLabel {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: #7b9aed;
    padding: 1px 0;
}
QLabel#sliderLabel {
    font-size: 11px;
    color: #8892b0;
    letter-spacing: 0.04em;
}
QLabel#valueLabel {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: #aac0ff;
}
QLabel#statusLabel {
    font-size: 10px;
    color: #556080;
    font-style: italic;
}
QLabel#helpText {
    color: #6070a0;
    font-size: 11px;
    line-height: 1.7;
}
QLabel:disabled {
    color: #6b738b;
}

/* ── ScrollBar ──────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0f1117;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2d3a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #3d5aad;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Status bar ─────────────────────────────────────────────── */
QStatusBar {
    background: #0c0e15;
    color: #d4d8e8;
    font-size: 11px;
    border-top: 1px solid #1e2130;
}
QStatusBar QLabel {
    color: #d4d8e8;
}

/* ── Menu bar ───────────────────────────────────────────────── */
QMenuBar {
    background: #0c0e15;
    color: #8892b0;
    border-bottom: 1px solid #1e2130;
    font-size: 12px;
}
QMenuBar::item:selected {
    background: #1e2130;
    color: #d4d8e8;
}
QMenu {
    background: #13161f;
    border: 1px solid #2a2d3a;
    color: #d4d8e8;
}
QMenu::item:selected {
    background: #1e3a6e;
    color: #e0eaff;
}
QMenu::item:disabled {
    color: #6b738b;
}
QMenu::separator {
    height: 1px;
    margin: 6px 10px;
    background: #3d5aad;
    border-top: 1px solid #6b738b;
}

/* ── Separator ──────────────────────────────────────────────── */
QFrame#separator {
    color: #1e2130;
    max-height: 1px;
}

/* ── Control panel scroll area ──────────────────────────────── */
QScrollArea {
    border: none;
    background: transparent;
}
"""


EXTRA_STYLESHEET = f"""
/* ── General inputs ────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE_ELEVATED_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-radius: 5px;
    padding: 4px 8px;
    padding-right: 24px;
    selection-background-color: {ACCENT_SOFT_COLOR.name()};
    selection-color: {TEXT_BRIGHT_COLOR.name()};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-top-right-radius: 5px;
    background-color: {SURFACE_COLOR.name()};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-bottom-right-radius: 5px;
    background-color: {SURFACE_COLOR.name()};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{SPINBOX_UP_ARROW_ICON}");
    width: 8px;
    height: 8px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{SPINBOX_DOWN_ARROW_ICON}");
    width: 8px;
    height: 8px;
}}
QLineEdit:read-only, QTextEdit:read-only, QPlainTextEdit:read-only {{
    background-color: {SURFACE_COLOR.name()};
    color: {TEXT_SECONDARY_COLOR.name()};
}}

/* ── Views and tables ──────────────────────────────────────── */
QTableView, QTreeView, QListView {{
    background-color: {BACKGROUND_COLOR.name()};
    alternate-background-color: {BACKGROUND_ALT_COLOR.name()};
    gridline-color: {SURFACE_BORDER_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
    selection-background-color: {ACCENT_SOFT_COLOR.name()};
    selection-color: {TEXT_BRIGHT_COLOR.name()};
    outline: none;
}}
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {{
    background-color: {ACCENT_SOFT_COLOR.name()};
    color: {TEXT_BRIGHT_COLOR.name()};
}}
QHeaderView::section {{
    background-color: {SURFACE_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
    border: none;
    border-bottom: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-right: 1px solid {SURFACE_BORDER_COLOR.name()};
    padding: 4px 8px;
    font-weight: 600;
}}

/* ── Toolbars / tool buttons / progress ───────────────────── */
QToolBar {{
    background: {BACKGROUND_ALT_COLOR.name()};
    border: none;
    border-bottom: 1px solid {SURFACE_BORDER_COLOR.name()};
    spacing: 4px;
    padding: 3px 4px;
    icon-size: 18px;
}}
QToolButton {{
    background: transparent;
    color: {TEXT_PRIMARY_COLOR.name()};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 6px;
}}
QToolButton:hover {{
    background: {SURFACE_COLOR.name()};
    border-color: {SURFACE_BORDER_COLOR.name()};
}}
QToolButton:checked {{
    background: {ACCENT_SOFT_COLOR.name()};
    border-color: {ACCENT_COLOR.name()};
}}
QProgressBar {{
    background: {BACKGROUND_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-radius: 4px;
    color: {TEXT_PRIMARY_COLOR.name()};
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_COLOR.name()};
    border-radius: 3px;
}}

/* ── Miscellaneous surfaces ───────────────────────────────── */
QGraphicsView {{
    background-color: {BACKGROUND_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
}}
QToolTip {{
    background-color: {SURFACE_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
    border: 1px solid {ACCENT_COLOR.name()};
    padding: 4px 6px;
}}
"""


APP_STYLESHEET = STYLESHEET + EXTRA_STYLESHEET


def build_stylesheet(scale_factor: float | None = None) -> str:
    return scale_qss(APP_STYLESHEET, scale_factor)


def build_dock_stylesheet(scale_factor: float | None = None) -> str:
    stylesheet = f"""
ads--CDockAreaTabBar {{
    background-color: {BACKGROUND_COLOR.name()};
    border-bottom: 1px solid {SURFACE_BORDER_COLOR.name()};
    height: 28px;
}}

ads--CDockWidgetTab {{
    background-color: {BACKGROUND_COLOR.name()};
    border: none;
    border-right: 1px solid {SURFACE_BORDER_COLOR.name()};
    border-top: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 3px 12px 2px 12px;
    color: {TEXT_SECONDARY_COLOR.name()};
    margin-top: 2px;
    font-weight: 700;
    min-width: 100px;
    max-width: 600px;
}}

ads--CDockWidgetTab:hover {{
    background-color: transparent;
    color: {TEXT_PRIMARY_COLOR.name()};
}}

ads--CDockWidgetTab[activeTab="true"] {{
    background-color: transparent;
    border-top: 1px solid transparent;
    border-left: 1px solid transparent;
    border-right: 1px solid transparent;
    border-bottom: 2px solid {ACCENT_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
    font-weight: 800;
    margin-top: 0px;
    padding: 4px 12px 3px 12px;
    min-width: 100px;
    max-width: 600px;
}}

ads--CDockWidgetTab[activeTab="true"]:hover {{
    background-color: transparent;
}}

ads--CDockWidgetTab QAbstractButton {{
    background-color: transparent;
    border: none;
    padding: 0px;
    margin: 0px 4px;
    width: 14px;
    height: 14px;
}}

ads--CDockWidgetTab QAbstractButton:hover {{
    background-color: rgba(255, 255, 255, 0.12);
    border-radius: 2px;
}}

ads--CDockWidgetTab:focus {{
    outline: none;
}}

ads--CDockArea {{
    background-color: {BACKGROUND_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
}}

/* ── Dock title-bar button icons ──────────────────────────────
   ADS ships extremely faint default glyphs for these buttons, so they
   read as blank squares against the dark theme. Override with explicit
   high-contrast icons for every spot ADS draws a button.

   Selector legend (ADS object names are stable across versions):
     - ads--CDockWidgetTab QToolButton  → small × on each tab
     - #detachGroupButton               → "make floating" arrow
     - #dockAreaCloseButton             → close the whole tab group
     - #dockAreaAutoHideButton          → toggle auto-hide
     - #floatingTitleCloseButton        → × on a floating window
     - #floatingTitleMaximizeButton     → maximize / restore on a floating window
*/
ads--CDockWidgetTab QToolButton {{
    qproperty-icon: url("{DOCK_CLOSE_ICON}");
    qproperty-iconSize: 12px 12px;
}}

ads--CDockAreaTitleBar QToolButton#detachGroupButton,
ads--CTitleBarButton#detachGroupButton {{
    qproperty-icon: url("{DOCK_DETACH_ICON}");
    qproperty-iconSize: 14px 14px;
}}

ads--CDockAreaTitleBar QToolButton#dockAreaCloseButton,
ads--CTitleBarButton#dockAreaCloseButton {{
    qproperty-icon: url("{DOCK_CLOSE_ICON}");
    qproperty-iconSize: 14px 14px;
}}

ads--CDockAreaTitleBar QToolButton#dockAreaAutoHideButton,
ads--CTitleBarButton#dockAreaAutoHideButton {{
    qproperty-icon: url("{DOCK_DETACH_ICON}");
    qproperty-iconSize: 14px 14px;
}}

ads--CFloatingWidgetTitleBar QToolButton#floatingTitleCloseButton {{
    qproperty-icon: url("{DOCK_CLOSE_ICON}");
    qproperty-iconSize: 14px 14px;
}}

ads--CFloatingWidgetTitleBar QToolButton#floatingTitleMaximizeButton {{
    qproperty-icon: url("{DOCK_MAXIMIZE_ICON}");
    qproperty-iconSize: 14px 14px;
}}

ads--CFloatingWidgetTitleBar QToolButton#floatingTitleMaximizeButton:checked {{
    qproperty-icon: url("{DOCK_RESTORE_ICON}");
    qproperty-iconSize: 14px 14px;
}}
"""
    return scale_qss(stylesheet, scale_factor)


def build_panel_stylesheet(scale_factor: float | None = None) -> str:
    stylesheet = f"""
QMenuBar {{
    background-color: {BACKGROUND_ALT_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
    border-bottom: 1px solid {SURFACE_BORDER_COLOR.name()};
}}
QMenuBar::item:selected {{
    background-color: {SURFACE_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
}}
QToolBar {{
    background-color: {BACKGROUND_ALT_COLOR.name()};
    border: none;
    padding: 2px;
    icon-size: 18px;
}}
QMenu {{
    background-color: {SURFACE_COLOR.name()};
    border: 1px solid {SURFACE_BORDER_COLOR.name()};
    color: {TEXT_PRIMARY_COLOR.name()};
}}
QMenu::item:selected {{
    background-color: {ACCENT_SOFT_COLOR.name()};
    color: {TEXT_BRIGHT_COLOR.name()};
}}
"""
    return scale_qss(stylesheet, scale_factor)


def apply_theme(app: QApplication) -> None:
    if get_scale_mode() == SCALE_MODE_AUTO:
        global _scale_factor
        _scale_factor = resolve_auto_scale_factor(app)

    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setFont(scale_font(APP_FONT_SIZE))
    app.setStyleSheet(build_stylesheet())