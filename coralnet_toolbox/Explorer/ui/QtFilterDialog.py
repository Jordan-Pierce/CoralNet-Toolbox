"""
Filter dialog for the Explorer annotation gallery.

One tab per filter in ANNOTATION_FILTERS (images, labels, types, status). Each
tab is a two column list, name and annotation count, and the selected rows are
the filter: click, Ctrl+click, Shift+click, drag and Ctrl+A select as in any
list. Counts are faceted: how many annotations a row would match given the
other tabs.

The dialog only *sets* the filter. The gallery's Apply Filter button is what
shows it.
"""

import fnmatch

from PyQt5.QtCore import Qt, QTimer, QItemSelection, QItemSelectionModel, pyqtSignal
from PyQt5.QtGui import QBrush, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QShortcut, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from coralnet_toolbox import theme as app_theme
from coralnet_toolbox.Explorer.core.annotation_filters import ANNOTATION_FILTERS, FacetIndex, FilterState
from coralnet_toolbox.Icons import get_window_icon


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


NAME_COLUMN = 0
COUNT_COLUMN = 1

VALUE_ROLE = Qt.UserRole          # on the name column: the option's value
CODE_ROLE = Qt.UserRole + 1       # on the name column: index into the option list
COUNT_ROLE = Qt.UserRole          # on the count column: the count as an int, for sorting


# ----------------------------------------------------------------------------------------------------------------------
# List
# ----------------------------------------------------------------------------------------------------------------------


class _FilterItem(QTreeWidgetItem):
    """Sorts names case-insensitively and counts as numbers."""

    def __lt__(self, other):
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else NAME_COLUMN
        if column == COUNT_COLUMN:
            return (self.data(COUNT_COLUMN, COUNT_ROLE) or 0) < (other.data(COUNT_COLUMN, COUNT_ROLE) or 0)
        return self.text(NAME_COLUMN).lower() < other.text(NAME_COLUMN).lower()


class FilterList(QTreeWidget):
    """Two column list (name, count) whose selected rows are the filter.

    Ctrl+A selects the rows shown by the search rather than every row, and adds
    to the selection, so rows picked under an earlier search stay picked.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAllColumnsShowFocus(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(NAME_COLUMN, QHeaderView.Stretch)
        header.setSectionResizeMode(COUNT_COLUMN, QHeaderView.ResizeToContents)

    def rows(self):
        return range(self.topLevelItemCount())

    def shown_rows(self):
        return [r for r in self.rows() if not self.topLevelItem(r).isHidden()]

    def set_rows_selected(self, rows, selected):
        """Select or deselect many rows with a single selectionChanged."""
        selection = QItemSelection()
        model = self.model()
        rows = sorted(rows)
        start = 0
        while start < len(rows):
            # One range per run of consecutive rows
            end = start
            while end + 1 < len(rows) and rows[end + 1] == rows[end] + 1:
                end += 1
            selection.select(model.index(rows[start], NAME_COLUMN), model.index(rows[end], COUNT_COLUMN))
            start = end + 1
        if selection.isEmpty():
            return
        command = QItemSelectionModel.Select if selected else QItemSelectionModel.Deselect
        self.selectionModel().select(selection, command | QItemSelectionModel.Rows)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.SelectAll):
            self.set_rows_selected(self.shown_rows(), True)
            event.accept()
            return
        super().keyPressEvent(event)


# ----------------------------------------------------------------------------------------------------------------------
# Tab
# ----------------------------------------------------------------------------------------------------------------------


class FilterTab(QWidget):
    """One filter: search box, quick buttons, the list, and a status line."""

    changed = pyqtSignal()

    def __init__(self, filt, options, spec, extra_buttons=(), current_value=None, parent=None):
        """
        Args:
            filt: The AnnotationFilter this tab edits.
            options: FilterOption list, one row each.
            spec: Currently picked values (frozenset) or None.
            extra_buttons: (text, tooltip, callable returning values to select).
            current_value: Value to mark as the open one (the current image).
        """
        super().__init__(parent)
        self.filt = filt
        self.options = list(options)

        layout = QVBoxLayout(self)

        # Search and quick buttons
        top_row = QHBoxLayout()
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText(f"Search {filt.title.lower()}, * and ? are wildcards")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip("Type to narrow the list. Press Enter to select every row shown.\n"
                                    "Rows selected under an earlier search stay selected.")
        self.search_edit.textChanged.connect(self._apply_search)
        self.search_edit.returnPressed.connect(self.select_shown)
        top_row.addWidget(self.search_edit, 1)

        for text, tooltip, values_getter in extra_buttons:
            top_row.addWidget(self._button(
                text, tooltip, lambda getter=values_getter: self.select_only_values(getter())))
        layout.addLayout(top_row)

        # The list
        self.list = FilterList(self)
        self.list.setHeaderLabels([filt.noun[0].capitalize(), "Annotations"])
        self.list.setToolTip("Click, Ctrl+click, Shift+click or drag to select. Ctrl+A selects every row shown.\n"
                             "Only annotations in the selected rows pass.")
        self._populate(spec, current_value)
        self.list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list, 1)

        self.status_label = QLabel(self)
        layout.addWidget(self.status_label)

        self._update_status()

    def _button(self, text, tooltip, callback):
        button = QPushButton(text, self)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda _=False: callback())
        return button

    def _populate(self, spec, current_value):
        """Fill the list; every row starts selected unless `spec` narrows it."""
        selected_rows = []
        for code, option in enumerate(self.options):
            item = _FilterItem([option.text, ""])
            item.setData(NAME_COLUMN, VALUE_ROLE, option.value)
            item.setData(NAME_COLUMN, CODE_ROLE, code)
            item.setToolTip(NAME_COLUMN, option.tooltip)
            item.setTextAlignment(COUNT_COLUMN, Qt.AlignRight | Qt.AlignVCenter)
            if current_value is not None and option.value == current_value:
                item.setText(NAME_COLUMN, f"{option.text}  (open)")
                item.setToolTip(NAME_COLUMN, f"{option.tooltip}\nOpen in the annotation window")
                font = item.font(NAME_COLUMN)
                font.setBold(True)
                item.setFont(NAME_COLUMN, font)
            self.list.addTopLevelItem(item)
        # Sort before selecting, so the rows selected are the rows shown
        self.list.setSortingEnabled(True)
        self.list.sortByColumn(NAME_COLUMN, Qt.AscendingOrder)
        for row in self.list.rows():
            if spec is None or self.list.topLevelItem(row).data(NAME_COLUMN, VALUE_ROLE) in spec:
                selected_rows.append(row)
        self.list.set_rows_selected(selected_rows, True)

    # -- State ---------------------------------------------------------------

    def selected_values(self):
        # From the selection model, not selectedItems(): that skips rows the
        # search hides, which are still selected and still part of the filter.
        return [index.data(VALUE_ROLE) for index in self.list.selectionModel().selectedRows(NAME_COLUMN)]

    def spec(self):
        """Selected values; None when every row is selected (no restriction)."""
        selected = self.selected_values()
        if len(selected) == len(self.options):
            return None
        return frozenset(selected)

    def reset(self):
        """Back to the default: every row selected, including rows the search hides."""
        self.list.set_rows_selected(self.list.rows(), True)

    def tab_title(self):
        selected = len(self.selected_values())
        return f"{self.filt.title} ({selected}/{len(self.options)})"

    def select_only_values(self, values):
        """Select exactly the rows holding `values`, shown or not."""
        wanted = set(values or ())
        if not wanted:
            return
        rows = [r for r in self.list.rows() if self.list.topLevelItem(r).data(NAME_COLUMN, VALUE_ROLE) in wanted]
        self.list.clearSelection()
        self.list.set_rows_selected(rows, True)

    def select_shown(self):
        self.list.set_rows_selected(self.list.shown_rows(), True)

    def deselect_shown(self):
        self.list.set_rows_selected(self.list.shown_rows(), False)

    def _on_selection_changed(self):
        self._update_status()
        self.changed.emit()

    # -- Counts and search ---------------------------------------------------

    def set_counts(self, counts):
        """Show each row's faceted count; rows nothing would match are dimmed."""
        muted = QBrush(app_theme.TEXT_MUTED_COLOR)
        normal = QBrush()
        # Re-sorting per row while counts change would be wasted work
        self.list.setSortingEnabled(False)
        try:
            for row in self.list.rows():
                item = self.list.topLevelItem(row)
                code = item.data(NAME_COLUMN, CODE_ROLE)
                count = int(counts[code]) if code < len(counts) else 0
                item.setText(COUNT_COLUMN, f"{count:,}")
                item.setData(COUNT_COLUMN, COUNT_ROLE, count)
                brush = normal if count else muted
                item.setForeground(NAME_COLUMN, brush)
                item.setForeground(COUNT_COLUMN, brush)
        finally:
            self.list.setSortingEnabled(True)

    def _apply_search(self, text):
        pattern = text.strip().lower()
        use_glob = any(ch in pattern for ch in "*?[")
        self.list.setUpdatesEnabled(False)
        try:
            for row in self.list.rows():
                item = self.list.topLevelItem(row)
                option = self.options[item.data(NAME_COLUMN, CODE_ROLE)]
                if not pattern:
                    visible = True
                elif use_glob:
                    visible = any(fnmatch.fnmatchcase(key, pattern) for key in option.keys)
                else:
                    visible = any(pattern in key for key in option.keys)
                item.setHidden(not visible)
        finally:
            self.list.setUpdatesEnabled(True)
        self._update_status()

    def _update_status(self):
        selected = len(self.selected_values())
        if selected == len(self.options):
            text = f"All {len(self.options):,} selected"
        elif selected:
            text = f"{selected:,} of {len(self.options):,} selected"
        else:
            text = f"Nothing selected, no {self.filt.noun[0]} passes"
        shown = len(self.list.shown_rows())
        if shown != len(self.options):
            text += f" · {shown:,} shown"
        self.status_label.setText(text)


# ----------------------------------------------------------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationFilterDialog(QDialog):
    """Modal dialog that edits a FilterState; one tab per filter.

    OK returns the new state through result_state(). Nothing is applied here.
    """

    RECOUNT_DELAY_MS = 150

    def __init__(self, main_window, annotations, state, current_image_path=None,
                 uncroppable=(), crop_threshold=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.crop_threshold = crop_threshold

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Filter Annotations")
        self.setObjectName("AnnotationFilterDialog")
        self.resize(app_theme.scale_int(520), app_theme.scale_int(520))

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.index = FacetIndex(annotations, main_window, uncroppable)
        finally:
            QApplication.restoreOverrideCursor()

        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget(self)
        self.tabs = []
        for filt in ANNOTATION_FILTERS:
            tab = FilterTab(
                filt, self.index.options[filt.key], state.get(filt.key),
                extra_buttons=self._extra_buttons(filt.key),
                current_value=current_image_path if filt.key == "image" else None,
                parent=self,
            )
            tab.changed.connect(self._schedule_recount)
            self.tabs.append(tab)
            self.tab_widget.addTab(tab, filt.title)
        layout.addWidget(self.tab_widget, 1)

        self.match_label = QLabel(self)
        self.match_label.setWordWrap(True)
        layout.addWidget(self.match_label)

        note = QLabel("OK sets the filter. Press Apply Filter in the gallery to show it.", self)
        note.setStyleSheet(f"color: {app_theme.TEXT_SECONDARY_COLOR.name()};")
        layout.addWidget(note)

        button_row = QHBoxLayout()
        reset_button = QPushButton("Reset", self)
        reset_button.setToolTip("Select every row on every tab (the default)")
        reset_button.clicked.connect(self._reset)
        button_row.addWidget(reset_button)
        separator = QFrame(self)
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        button_row.addWidget(separator)
        all_button = QPushButton("All", self)
        all_button.setToolTip("Select every row shown on this tab (Ctrl+A)")
        all_button.clicked.connect(lambda: self.tab_widget.currentWidget().select_shown())
        button_row.addWidget(all_button)
        none_button = QPushButton("None", self)
        none_button.setToolTip("Deselect every row shown on this tab, to pick from scratch")
        none_button.clicked.connect(lambda: self.tab_widget.currentWidget().deselect_shown())
        button_row.addWidget(none_button)
        button_row.addStretch(1)
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        ok_button = QPushButton("OK", self)
        ok_button.setToolTip("Set this filter (Ctrl+Enter)")
        ok_button.clicked.connect(self.accept)
        button_row.addWidget(ok_button)
        layout.addLayout(button_row)

        # Enter belongs to the search boxes (select every row shown), so no
        # button may be the dialog default. Ctrl+Enter is OK instead.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        for sequence in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(sequence), self, activated=self.accept)

        self._recount_timer = QTimer(self)
        self._recount_timer.setSingleShot(True)
        self._recount_timer.setInterval(self.RECOUNT_DELAY_MS)
        self._recount_timer.timeout.connect(self._recount)

        self._recount()

    def _extra_buttons(self, key):
        if key != "image":
            return ()
        return (
            ("Current", "Select only the image open in the annotation window", self._current_image_paths),
            ("Highlighted", "Select only the images highlighted in the Image window", self._highlighted_paths),
        )

    def _current_image_paths(self):
        path = getattr(getattr(self.main_window, "annotation_window", None), "current_image_path", None)
        return [path] if path else []

    def _highlighted_paths(self):
        try:
            return self.main_window.image_window.table_model.get_highlighted_paths()
        except Exception:
            return []

    def _schedule_recount(self):
        self._recount_timer.start()

    def _reset(self):
        """Select every row on every tab, recounting once rather than per tab."""
        for tab in self.tabs:
            tab.blockSignals(True)
            try:
                tab.reset()
            finally:
                tab.blockSignals(False)
        self._recount()

    def result_state(self):
        state = FilterState()
        for filt, tab in zip(ANNOTATION_FILTERS, self.tabs):
            state.set(filt.key, tab.spec())
        return state

    def _recount(self):
        self._recount_timer.stop()
        counts = self.index.count(self.result_state())
        for i, (filt, tab) in enumerate(zip(ANNOTATION_FILTERS, self.tabs)):
            tab.set_counts(counts.per_key[filt.key])
            self.tab_widget.setTabText(i, tab.tab_title())

        text = (f"{counts.total:,} of {self.index.n:,} annotations match, "
                f"in {counts.images:,} image{'s' if counts.images != 1 else ''}.")
        warn = False
        if counts.crops:
            text += f" {counts.crops:,} still need cropping."
            if self.crop_threshold is not None and counts.crops > self.crop_threshold:
                text += " Applying will ask before cropping that many."
                warn = True
        self.match_label.setText(text)
        self.match_label.setStyleSheet(f"color: {app_theme.ATTENTION_COLOR.name()};" if warn else "")
