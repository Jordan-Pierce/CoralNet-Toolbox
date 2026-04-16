from PyQt5 import QtCore, QtWidgets


class _PersistentMenu(QtWidgets.QMenu):
    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.pos())
        owner = self.parentWidget()

        if action is not None and action.isEnabled() and owner is not None and hasattr(owner, "_handle_menu_action"):
            owner._handle_menu_action(action)
            self.setActiveAction(action)

        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            owner = self.parentWidget()
            action = self.activeAction()
            if action is not None and owner is not None and hasattr(owner, "_handle_menu_action"):
                owner._handle_menu_action(action)
                event.accept()
                return

        super().keyPressEvent(event)


class MultiSelectCombo(QtWidgets.QWidget):
    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self, options=None, parent=None):
        super().__init__(parent)
        self._options = []
        self._actions = []

        self.button = QtWidgets.QToolButton(self)
        self.button.setText('All')
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = _PersistentMenu(self)
        self.button.setMenu(self.menu)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)

        if options:
            self.set_options(options)

    def set_options(self, options):
        """options: list of (display_text, value)"""
        self._options = list(options)
        self.menu.clear()
        self._actions = []

        # Ensure 'All' at top
        all_action = QtWidgets.QAction('All', self.menu)
        all_action.setCheckable(True)
        all_action.setChecked(True)
        self.menu.addAction(all_action)
        self._actions.append((all_action, None))

        for text, value in self._options:
            act = QtWidgets.QAction(text, self.menu)
            act.setCheckable(True)
            act.setData(value)
            self.menu.addAction(act)
            self._actions.append((act, value))

        self._update_button_text()

    def _handle_menu_action(self, action):
        if not self._actions:
            return

        all_action = self._actions[0][0]

        if action is all_action:
            self._select_all()
            return

        action.setChecked(not action.isChecked())

        any_checked = any(act.isChecked() for act, val in self._actions[1:])
        all_action.setChecked(not any_checked)

        self._update_button_text()
        self.selection_changed.emit(self.selected_values())

    def _select_all(self):
        if not self._actions:
            return

        all_action = self._actions[0][0]
        all_action.setChecked(True)
        for act, val in self._actions[1:]:
            act.setChecked(False)

        self._update_button_text()
        self.selection_changed.emit(self.selected_values())

    # Compatibility helpers to mimic QComboBox API for legacy callers
    def currentData(self):
        """Return a single representative data value or 'all' when all selected."""
        vals = self.selected_values()
        if vals is None:
            return 'all'
        if isinstance(vals, (list, tuple)) and len(vals) == 1:
            return vals[0]
        # return first selected as representative
        return vals[0] if vals else None

    def currentText(self):
        vals = self.selected_values()
        if vals is None:
            return 'All'
        if isinstance(vals, (list, tuple)):
            if len(vals) > 2:
                return f"{len(vals)} selected"
            return ', '.join(str(v) for v in vals)
        return str(vals)

    def clear(self):
        # reset to All
        if not self._actions:
            return
        self._select_all()

    def _update_button_text(self):
        vals = self.selected_values()
        if vals is None:
            self.button.setText('All')
        else:
            # show up to 2 items
            if len(vals) > 2:
                self.button.setText(f"{len(vals)} selected")
            else:
                self.button.setText(', '.join(str(v) for v in vals))

    def selected_values(self):
        # None represents All
        if not self._actions:
            return None
        # if All checked or none checked -> None
        if self._actions[0][0].isChecked():
            return None
        selected = [act.data() for act, val in self._actions[1:] if act.isChecked()]
        if not selected:
            return None
        return selected
