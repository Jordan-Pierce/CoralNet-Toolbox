"""Compatibility shim: the timer was renamed from `QtTimer`/`TimerGroupBox` to
`QtTimerWindow`/`TimerWindow`. Import the new implementation and re-export
the old names so existing imports keep working.

This file intentionally contains only re-exports.
"""

from coralnet_toolbox.QtTimerWindow import (
    TimerWorker,
    TimerWidget,
    TimerWindow,
)

# Backwards compatibility: export the old class name
TimerGroupBox = TimerWindow

__all__ = [
    'TimerWorker',
    'TimerWidget',
    'TimerWindow',
    'TimerGroupBox',
]
