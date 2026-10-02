"""Notices shown to the user once the main window is up.

For now this is the Python version notice: environments on anything other than
the recommended Python get a message box at startup until they upgrade or tick
"Don't show this again", and the same message goes to the console on every
launch regardless.

Called from main.run() after the window is shown, not from MainWindow's
constructor: the smoke tests build MainWindow directly, and a modal dialog there
would hang them.
"""

import json
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QMessageBox

from coralnet_toolbox.paths import app_home

RECOMMENDED_PYTHON = (3, 12)
INSTALL_URL = "https://jordan-pierce.github.io/CoralNet-Toolbox/installation"

# Upstream end of life for the older versions still allowed by pyproject.toml
PYTHON_END_OF_LIFE = {(3, 10): "October 2026", (3, 11): "October 2027"}

SETTINGS_FILE = "settings.json"
DISMISSED_KEY = "python_notice_dismissed"


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def python_version_message(version_info=sys.version_info):
    """The notice for this Python, or None if it is the recommended one.

    Args:
        version_info: A sys.version_info-like tuple.

    Returns:
        str | None: Plain-text message.
    """
    current = tuple(version_info[:2])
    if current == RECOMMENDED_PYTHON:
        return None

    running = ".".join(str(part) for part in version_info[:3])
    recommended = ".".join(str(part) for part in RECOMMENDED_PYTHON)

    if current > RECOMMENDED_PYTHON:
        return (f"This environment runs Python {running}. The toolbox is tested on Python "
                f"{recommended}, and some of its dependencies may not support {running} yet. "
                f"If something fails to install or start, use Python {recommended}.\n\n"
                f"Installation guide: {INSTALL_URL}")

    end_of_life = PYTHON_END_OF_LIFE.get(current)
    reason = (f"Python {current[0]}.{current[1]} stops receiving security fixes in {end_of_life}, and "
              if end_of_life else "")
    return (f"This environment runs Python {running}. Python {recommended} is recommended: "
            f"{reason}newer releases of some dependencies already require a newer Python.\n\n"
            f"Python cannot be upgraded inside an existing environment. Create a new environment "
            f"with Python {recommended} and install the toolbox there.\n\n"
            f"Installation guide: {INSTALL_URL}")


def _settings_path():
    return app_home() / SETTINGS_FILE


def _load_settings():
    """The settings file's contents, or {} if it is missing or unreadable."""
    try:
        with open(_settings_path(), 'r', encoding='utf-8') as file:
            settings = json.load(file)
        return settings if isinstance(settings, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(settings):
    """Write the settings file; best effort, a failure only means the notice returns."""
    try:
        with open(_settings_path(), 'w', encoding='utf-8') as file:
            json.dump(settings, file, indent=4)
    except OSError as e:
        print(f"Could not save {_settings_path()}: {e}")


def show_python_version_notice(parent=None):
    """Print the Python version notice, and show it unless dismissed.

    The console always gets it. The dialog is skipped when the user ticked
    "Don't show this again" for this Python version, and inside the Docker
    image, whose Python the user cannot change.

    Args:
        parent (QWidget, optional): Parent for the message box.
    """
    message = python_version_message()
    if message is None:
        return

    print(f"\nNOTE: {message}\n")

    if os.path.exists("/.dockerenv"):
        return

    current = f"{sys.version_info[0]}.{sys.version_info[1]}"
    settings = _load_settings()
    if settings.get(DISMISSED_KEY) == current:
        return

    link = f'<a href="{INSTALL_URL}">{INSTALL_URL}</a>'
    html = message.replace(INSTALL_URL, link).replace("\n", "<br>")

    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Python Version")
    msg_box.setTextFormat(Qt.RichText)
    msg_box.setText(html)
    msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse |
                                    Qt.TextSelectableByKeyboard |
                                    Qt.LinksAccessibleByMouse)
    check_box = QCheckBox("Don't show this again")
    msg_box.setCheckBox(check_box)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()

    if check_box.isChecked():
        settings[DISMISSED_KEY] = current
        _save_settings(settings)


def show_startup_notices(parent=None):
    """Everything to tell the user once the main window is up.

    Never raises: a notice that fails is printed and skipped rather than
    interrupting startup.

    Args:
        parent (QWidget, optional): Parent for any dialogs.
    """
    try:
        show_python_version_notice(parent)
    except Exception as e:
        print(f"Startup notice failed: {e}")
