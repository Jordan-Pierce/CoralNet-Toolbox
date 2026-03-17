from PyQt5 import QtCore, QtWidgets


class MultiSelectCombo(QtWidgets.QWidget):
    selection_changed = QtCore.pyqtSignal(object)

    def __init__(self, options=None, parent=None):
        super().__init__(parent)
        self._options = []
        self._actions = []

        self.button = QtWidgets.QToolButton(self)
        self.button.setText('All')
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.menu = QtWidgets.QMenu(self)
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
        all_action.triggered.connect(self._on_all_toggled)
        self.menu.addAction(all_action)
        self._actions.append((all_action, None))

        for text, value in self._options:
            act = QtWidgets.QAction(text, self.menu)
            act.setCheckable(True)
            act.setData(value)
            act.toggled.connect(self._on_option_toggled)
            self.menu.addAction(act)
            self._actions.append((act, value))

        self._update_button_text()

    def _on_all_toggled(self, checked):
        if checked:
            # uncheck all others
            for act, val in self._actions[1:]:
                act.setChecked(False)
        self._update_button_text()
        self.selection_changed.emit(self.selected_values())

    def _on_option_toggled(self, checked):
        # if any option checked, uncheck All
        any_checked = any(act.isChecked() for act, val in self._actions[1:])
        self._actions[0][0].setChecked(not any_checked)
        self._update_button_text()
        self.selection_changed.emit(self.selected_values())

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
