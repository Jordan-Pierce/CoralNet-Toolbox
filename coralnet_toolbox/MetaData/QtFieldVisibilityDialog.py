import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QListWidget, QListWidgetItem, QPushButton, QAbstractItemView)

from coralnet_toolbox.Icons import get_icon, get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FieldVisibilityDialog(QDialog):
    """Choose which custom metadata fields are shown, and in what order.

    A dual transfer list: fields move between Hidden and Visible, and their
    position within Visible is their display order in the panel. Ordering lives
    here rather than as up/down buttons on the panel toolbar, so arranging a
    schema is one focused task instead of many round trips.

    Built-in and Raw Data rows are deliberately absent -- they are always shown
    and are computed per annotation rather than defined by the schema, so
    listing them would only take up space.
    """

    def __init__(self, schema, parent=None):
        """Initialize the dialog for the given schema."""
        super().__init__(parent)
        self.schema = schema

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle("Show / Hide Metadata Fields")
        self.setObjectName("FieldVisibilityDialog")
        self.resize(560, 460)

        self.layout = QVBoxLayout(self)

        self.setup_info_layout()
        self.setup_transfer_layout()
        self.setup_buttons_layout()

        self.populate()
        self.update_button_states()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def setup_info_layout(self):
        """Informational header explaining what the dialog does."""
        group_box = QGroupBox("About")
        layout = QVBoxLayout()

        info_label = QLabel(
            "Move fields between <b>Hidden</b> and <b>Visible</b>, and use the up and down "
            "arrows to set the order they appear in the Metadata panel.<br><br>"
            "Hiding a field only affects the display. No stored values are changed or "
            "deleted, and hidden fields are still saved with the project.<br><br>"
            "Built-in fields are always shown and are not listed here."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_transfer_layout(self):
        """Set up the Hidden | buttons | Visible transfer lists."""
        group_box = QGroupBox("Fields")
        outer = QHBoxLayout()

        # --- Hidden side ---
        hidden_column = QVBoxLayout()
        hidden_column.addWidget(QLabel("Hidden"))
        self.hidden_list = QListWidget()
        self.hidden_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.hidden_list.setToolTip("Fields that are not shown in the Metadata panel.")
        self.hidden_list.itemSelectionChanged.connect(self.update_button_states)
        self.hidden_list.itemDoubleClicked.connect(lambda _item: self.move_to_visible())
        hidden_column.addWidget(self.hidden_list)
        outer.addLayout(hidden_column)

        # --- Transfer buttons ---
        transfer_column = QVBoxLayout()
        transfer_column.addStretch()

        self.show_button = QPushButton()
        self.show_button.setIcon(get_icon("right_chevron.svg"))
        self.show_button.setToolTip("Show the selected fields")
        self.show_button.clicked.connect(self.move_to_visible)
        transfer_column.addWidget(self.show_button)

        self.hide_button = QPushButton()
        self.hide_button.setIcon(get_icon("left_chevron.svg"))
        self.hide_button.setToolTip("Hide the selected fields")
        self.hide_button.clicked.connect(self.move_to_hidden)
        transfer_column.addWidget(self.hide_button)

        transfer_column.addSpacing(16)

        self.up_button = QPushButton()
        self.up_button.setIcon(get_icon("up_chevron.svg"))
        self.up_button.setToolTip("Move the selected fields up")
        self.up_button.clicked.connect(lambda: self.move_within_visible(-1))
        transfer_column.addWidget(self.up_button)

        self.down_button = QPushButton()
        self.down_button.setIcon(get_icon("down_chevron.svg"))
        self.down_button.setToolTip("Move the selected fields down")
        self.down_button.clicked.connect(lambda: self.move_within_visible(1))
        transfer_column.addWidget(self.down_button)

        transfer_column.addStretch()
        outer.addLayout(transfer_column)

        # --- Visible side ---
        visible_column = QVBoxLayout()
        visible_column.addWidget(QLabel("Visible (in display order)"))
        self.visible_list = QListWidget()
        self.visible_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.visible_list.setToolTip("Fields shown in the Metadata panel, top to bottom.")
        self.visible_list.itemSelectionChanged.connect(self.update_button_states)
        self.visible_list.itemDoubleClicked.connect(lambda _item: self.move_to_hidden())
        visible_column.addWidget(self.visible_list)
        outer.addLayout(visible_column)

        group_box.setLayout(outer)
        self.layout.addWidget(group_box)

    def setup_buttons_layout(self):
        """Set up the OK / Cancel row."""
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def make_item(self, field):
        """Build a list row carrying a field's name."""
        item = QListWidgetItem(field.label)
        item.setData(Qt.UserRole, field.name)
        if field.description:
            item.setToolTip(field.description)
        return item

    def populate(self):
        """Fill both lists from the schema, preserving its current order."""
        self.hidden_list.clear()
        self.visible_list.clear()

        for field in self.schema:
            item = self.make_item(field)
            if field.visible:
                self.visible_list.addItem(item)
            else:
                self.hidden_list.addItem(item)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def _take_selected(self, source):
        """Remove and return the selected rows from a list, top to bottom."""
        rows = sorted(source.row(item) for item in source.selectedItems())
        return [source.takeItem(row) for row in reversed(rows)][::-1]

    def move_to_visible(self):
        """Show the fields selected on the hidden side."""
        for item in self._take_selected(self.hidden_list):
            self.visible_list.addItem(item)
            item.setSelected(True)
        self.update_button_states()

    def move_to_hidden(self):
        """Hide the fields selected on the visible side."""
        for item in self._take_selected(self.visible_list):
            self.hidden_list.addItem(item)
            item.setSelected(True)
        self.update_button_states()

    def move_within_visible(self, offset):
        """Move the selected visible fields up or down one position."""
        rows = sorted(self.visible_list.row(item) for item in self.visible_list.selectedItems())
        if not rows:
            return

        # Walk in the direction of travel so items cannot leapfrog each other,
        # and stop at the ends rather than wrapping around.
        if offset < 0:
            if rows[0] + offset < 0:
                return
            order = rows
        else:
            if rows[-1] + offset >= self.visible_list.count():
                return
            order = list(reversed(rows))

        for row in order:
            item = self.visible_list.takeItem(row)
            self.visible_list.insertItem(row + offset, item)
            item.setSelected(True)

        self.update_button_states()

    def update_button_states(self):
        """Enable only the transfers that are currently possible."""
        hidden_selected = bool(self.hidden_list.selectedItems())
        visible_selected = bool(self.visible_list.selectedItems())

        self.show_button.setEnabled(hidden_selected)
        self.hide_button.setEnabled(visible_selected)

        rows = sorted(self.visible_list.row(item) for item in self.visible_list.selectedItems())
        self.up_button.setEnabled(bool(rows) and rows[0] > 0)
        self.down_button.setEnabled(bool(rows) and rows[-1] < self.visible_list.count() - 1)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def apply(self):
        """Write visibility and display order back onto the schema."""
        ordered = []

        for index in range(self.visible_list.count()):
            name = self.visible_list.item(index).data(Qt.UserRole)
            field = self.schema.get_field(name)
            if field is not None:
                field.visible = True
                ordered.append(field)

        for index in range(self.hidden_list.count()):
            name = self.hidden_list.item(index).data(Qt.UserRole)
            field = self.schema.get_field(name)
            if field is not None:
                field.visible = False
                ordered.append(field)

        # Visible fields lead in the order the user arranged; hidden ones keep
        # their relative order behind them so unhiding is predictable.
        self.schema.fields = ordered
