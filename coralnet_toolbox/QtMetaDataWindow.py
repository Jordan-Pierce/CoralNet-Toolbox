import warnings

from PyQt5.QtCore import Qt, QDate, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QToolBar, QSizePolicy, QTreeWidget, QTreeWidgetItem,
                             QAbstractItemView, QHeaderView, QCheckBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QPlainTextEdit, QDateEdit,
                             QListWidget, QListWidgetItem, QApplication, QMessageBox)

from coralnet_toolbox.MetaData import MetaDataSchema
from coralnet_toolbox.MetaData.QtBuiltInFields import compute_builtin_fields
from coralnet_toolbox.MetaData.QtBuiltInFields import format_unconvertible_note
from coralnet_toolbox.MetaData.QtFieldDialog import AddFieldDialog
from coralnet_toolbox.MetaData.QtFieldDialog import EditFieldDialog
from coralnet_toolbox.MetaData.QtFieldVisibilityDialog import FieldVisibilityDialog
from coralnet_toolbox.MetaData.QtConfirmDialog import confirm_delete_field
from coralnet_toolbox.MetaData.QtConfirmDialog import confirm_edit_field
from coralnet_toolbox.MetaData.QtConfirmDialog import report_promotion

from coralnet_toolbox.QtProgressBar import ProgressBar
from coralnet_toolbox.Icons import get_icon
from coralnet_toolbox import theme as app_theme

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Shown in an editor when the selected annotations disagree about a value.
# Writing is suppressed until the user actively changes the widget, so merely
# viewing a mixed selection never overwrites anything.
MULTIPLE_TEXT = "<multiple values>"

# Shown for computed built-in rows when more than one annotation is selected.
# The rows stay put so the panel keeps its shape instead of collapsing.
MULTIPLE_SELECTION_TEXT = "<multiple annotations selected>"

# Live edits are committed on this delay, so holding a key does not trigger one
# write per keystroke across a large selection.
COMMIT_DELAY_MS = 200

# Above this many annotations, a bulk operation gets a progress bar.
PROGRESS_THRESHOLD = 200


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MetaDataWindow(QWidget):
    """Dock showing structured metadata for the selected vector annotation(s).

    Three tiers are displayed:
      - Custom: user-defined typed fields governed by the project's
        MetaDataSchema. Editable, and editable across a multi-selection.
      - Built-in: values derived from geometry and the raster. Read-only, and
        recomputed on every refresh so they can never go stale.
      - Raw Data: whatever remains in annotation.data that could not be
        promoted into a real field (nested structures). Read-only.
    """

    schemaChanged = pyqtSignal()

    def __init__(self, main_window, parent=None):
        """Initialize the MetaDataWindow widget."""
        super().__init__(parent)
        self.main_window = main_window
        self.annotation_window = main_window.annotation_window

        # The project's field definitions. Replaced wholesale on project load.
        self.schema = MetaDataSchema()

        # Vector annotations currently selected, in selection order.
        self.annotations = []

        # field name -> editor widget, rebuilt with the tree
        self._editors = {}
        # Suppresses write-back while editors are being populated programmatically.
        self._populating = False
        # Annotations whose annotationUpdated signal we are currently connected to.
        self._connected = []
        # Group expansion is remembered across rebuilds, including while a group
        # is absent -- an empty Raw Data group must not forget it was open.
        self._expanded = {'Built-in': False, 'Raw Data': False, 'Custom': True}
        # Field with an edit not yet written, and the timer that will write it.
        self._pending_field = None
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(COMMIT_DELAY_MS)
        self._commit_timer.timeout.connect(self.flush_pending)

        self.setup_ui()

        # The AnnotationManager relays every selection change from the canvas and
        # from the explorer windows, so subscribing here covers all of them.
        self.main_window.annotation_manager.selectionChanged.connect(self.on_selection_changed)

        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def setup_ui(self):
        """Set up the user interface. The payload is ONLY the tree."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Instantiate toolbar widgets here so __init__ can connect signals,
        # but do NOT put them in the main layout -- the dock mounts them.
        self._init_action_widgets()
        self.setup_tree_section()

    def _init_action_widgets(self):
        """Instantiate action buttons and the filter bar."""
        self.add_field_button = QPushButton()
        self.add_field_button.setIcon(get_icon("add.svg"))
        self.add_field_button.setIconSize(app_theme.scale_size(16))
        self.add_field_button.setToolTip("Add Field")
        self.add_field_button.clicked.connect(self.open_add_field_dialog)

        self.delete_field_button = QPushButton()
        self.delete_field_button.setIcon(get_icon("remove.svg"))
        self.delete_field_button.setIconSize(app_theme.scale_size(16))
        self.delete_field_button.setToolTip("Delete Field")
        self.delete_field_button.setEnabled(False)
        self.delete_field_button.clicked.connect(self.delete_selected_field)

        self.edit_field_button = QPushButton()
        self.edit_field_button.setIcon(get_icon("edit.svg"))
        self.edit_field_button.setIconSize(app_theme.scale_size(16))
        self.edit_field_button.setToolTip("Edit Field")
        self.edit_field_button.setEnabled(False)
        self.edit_field_button.clicked.connect(self.open_edit_field_dialog)

        self.visibility_button = QPushButton()
        self.visibility_button.setIcon(get_icon("eye.svg"))
        self.visibility_button.setIconSize(app_theme.scale_size(16))
        self.visibility_button.setToolTip("Show / Hide Fields and Set Their Order")
        self.visibility_button.clicked.connect(self.open_visibility_dialog)

        self.adopt_button = QPushButton()
        self.adopt_button.setIcon(get_icon("magic.svg"))
        self.adopt_button.setIconSize(app_theme.scale_size(16))
        self.adopt_button.setToolTip("Adopt Fields from Imported Data")
        self.adopt_button.clicked.connect(self.adopt_from_data)

        self.filter_bar = QLineEdit()
        self.filter_bar.setPlaceholderText("Filter Fields")
        self.filter_bar.textChanged.connect(self.rebuild)

    def setup_tree_section(self):
        """Set up the core payload: the property grid."""
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Field", "Value"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.header().setStretchLastSection(True)
        self.tree.setColumnWidth(0, app_theme.scale_int(150))
        self.tree.itemSelectionChanged.connect(self.update_button_states)

        self.layout.addWidget(self.tree)

    # --- DOCK WRAPPER HOOKS ---

    def create_action_toolbar(self) -> QToolBar:
        """Create the first row top toolbar containing action buttons."""
        toolbar = QToolBar("Metadata Actions")
        toolbar.setMovable(False)

        button_container = QWidget()
        container_layout = QHBoxLayout(button_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)

        container_layout.addWidget(self.add_field_button)
        container_layout.addWidget(self.delete_field_button)
        container_layout.addWidget(self.edit_field_button)
        container_layout.addWidget(self.visibility_button)
        container_layout.addWidget(self.adopt_button)

        button_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(button_container)

        return toolbar

    def create_filter_toolbar(self) -> QToolBar:
        """Create the second row top toolbar containing the filter search bar."""
        toolbar = QToolBar("Metadata Filter")
        toolbar.setMovable(False)

        self.filter_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self.filter_bar)

        return toolbar

    def refresh_scaling(self):
        """Refresh icon and control sizes after a UI scale change."""
        for button in (self.add_field_button, self.delete_field_button, self.edit_field_button,
                       self.visibility_button, self.adopt_button):
            button.setIconSize(app_theme.scale_size(16))

        self.tree.setColumnWidth(0, app_theme.scale_int(150))

    # ------------------------------------------------------------------
    # Schema access
    # ------------------------------------------------------------------

    def set_schema(self, schema):
        """Replace the project's schema wholesale (used on project load)."""
        self.schema = schema or MetaDataSchema()
        self.rebuild()
        self.schemaChanged.emit()

    def get_all_annotations(self):
        """Return every vector annotation in the project.

        Mask annotations are excluded: they are a per-image label raster, not a
        discrete object that metadata could describe.
        """
        return [annotation for annotation in self.annotation_window.annotations_dict.values()
                if not getattr(annotation, 'is_mask_annotation', False)]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def on_selection_changed(self, annotation_ids):
        """Handle a change in the canvas/explorer annotation selection."""
        # Write any half-typed edit against the OUTGOING selection first --
        # otherwise cycling annotations by hotkey silently discards it.
        self.flush_pending()

        annotations_dict = self.annotation_window.annotations_dict
        annotations = []
        for annotation_id in (annotation_ids or []):
            annotation = annotations_dict.get(annotation_id)
            # Mask annotations have no per-object identity to describe.
            if annotation is not None and not getattr(annotation, 'is_mask_annotation', False):
                annotations.append(annotation)

        self.annotations = annotations
        self._reconnect_annotation_signals()
        self.rebuild()

    def _reconnect_annotation_signals(self):
        """Track annotationUpdated on exactly the current selection.

        Built-in values are derived from geometry, so an edit elsewhere in the
        app must redraw them. Connections are dropped as soon as an annotation
        leaves the selection, mirroring how AnnotationWindow manages the
        equivalent ConfidenceWindow connections.
        """
        for annotation in self._connected:
            try:
                annotation.annotationUpdated.disconnect(self.on_annotation_updated)
            except TypeError:
                pass

        self._connected = list(self.annotations)
        for annotation in self._connected:
            try:
                annotation.annotationUpdated.connect(self.on_annotation_updated)
            except (TypeError, AttributeError):
                pass

    def on_annotation_updated(self, annotation):
        """Refresh when a selected annotation changes underneath us."""
        if annotation in self.annotations:
            self.rebuild()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def refresh(self):
        """Rebuild the grid from the current schema and selection."""
        self.rebuild()

    def rebuild(self):
        """Rebuild the property grid."""
        # Remember which field was selected so a rebuild triggered by an edit
        # does not lose the user's place.
        self._capture_expanded_state()
        selected_field = self.get_selected_field_name()

        self._populating = True
        try:
            self.tree.clear()
            self._editors = {}

            # Read-only context first, then the fields the user actually edits.
            self._build_builtin_group()
            self._build_raw_data_group()
            self._build_custom_group()

            self._restore_expanded_state()
            if selected_field:
                self.select_field(selected_field)
        finally:
            self._populating = False

        self.update_button_states()

    def _capture_expanded_state(self):
        """Record which top-level groups are expanded.

        Updates in place rather than returning a snapshot: a group that is not
        currently built must keep whatever state the user last set, instead of
        being forgotten and re-collapsing next time it appears.
        """
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            self._expanded[item.text(0).split(' (')[0]] = item.isExpanded()

    def _restore_expanded_state(self):
        """Re-apply the remembered expansion state."""
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            title = item.text(0).split(' (')[0]
            item.setExpanded(self._expanded.get(title, title == 'Custom'))

    def _make_group(self, title, count):
        """Create a top-level group row."""
        item = QTreeWidgetItem(self.tree, [f"{title} ({count})", ""])
        item.setFirstColumnSpanned(True)
        item.setFlags(Qt.ItemIsEnabled)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

    def _matches_filter(self, text):
        """Return True if a row label passes the filter bar."""
        needle = self.filter_bar.text().strip().lower()
        return not needle or needle in str(text).lower()

    def _build_custom_group(self):
        """Build the editable, schema-governed tier."""
        fields = [field for field in self.schema.visible_fields()
                  if self._matches_filter(field.label) or self._matches_filter(field.name)]

        group = self._make_group("Custom", len(fields))
        if not fields:
            return

        has_selection = bool(self.annotations)

        for field in fields:
            item = QTreeWidgetItem(group, [field.label, ""])
            item.setData(0, Qt.UserRole, field.name)
            if field.description:
                item.setToolTip(0, field.description)

            value, is_multiple = self._collect_value(field)
            editor = self._create_editor(field, value, is_multiple)
            editor.setEnabled(has_selection)
            self._editors[field.name] = editor
            self.tree.setItemWidget(item, 1, editor)

    def _build_builtin_group(self):
        """Build the computed, read-only tier.

        With several annotations selected the values are per-annotation and so
        cannot be shown, but the rows stay in place: keeping the panel's shape
        steady is less jarring than the whole group vanishing and reappearing.
        The structure is taken from the first selected annotation.
        """
        if not self.annotations:
            self._make_group("Built-in", 0)
            return

        multiple = len(self.annotations) > 1
        fields, unconvertible_units = compute_builtin_fields(self.annotations[0], self.main_window)
        rows = [(name, value) for name, value in fields.items() if self._matches_filter(name)]

        group = self._make_group("Built-in", len(rows))
        for name, value in rows:
            text = MULTIPLE_SELECTION_TEXT if multiple else str(value)
            child = QTreeWidgetItem(group, [name, text])
            child.setToolTip(1, text)
            if multiple:
                child.setForeground(1, app_theme.TEXT_MUTED_COLOR)

        if not multiple:
            note = format_unconvertible_note(unconvertible_units)
            if note:
                child = QTreeWidgetItem(group, ["Note", note])
                child.setToolTip(1, note)

    def _build_raw_data_group(self):
        """Build the residual annotation.data tier.

        Always built, even when empty, so the group keeps its position and its
        expanded state as the user moves between annotations.
        """
        rows = []
        multiple = len(self.annotations) > 1

        if len(self.annotations) == 1:
            data = getattr(self.annotations[0], 'data', None) or {}
            rows = [(key, value) for key, value in data.items() if self._matches_filter(key)]
        elif multiple:
            # Only the keys every selected annotation shares are meaningful.
            common = None
            for annotation in self.annotations:
                keys = set((getattr(annotation, 'data', None) or {}).keys())
                common = keys if common is None else (common & keys)
            rows = [(key, MULTIPLE_SELECTION_TEXT) for key in sorted(common or ())
                    if self._matches_filter(key)]

        group = self._make_group("Raw Data", len(rows))
        for key, value in rows:
            child = QTreeWidgetItem(group, [str(key), str(value)])
            child.setToolTip(1, str(value))
            if multiple:
                child.setForeground(1, app_theme.TEXT_MUTED_COLOR)

    # ------------------------------------------------------------------
    # Editors
    # ------------------------------------------------------------------

    def _collect_value(self, field):
        """Return (value, is_multiple) for a field across the selection."""
        if not self.annotations:
            return field.default, False

        values = [self.schema.get_value(annotation, field.name) for annotation in self.annotations]
        first = values[0]
        for value in values[1:]:
            if value != first:
                return first, True
        return first, False

    def _create_editor(self, field, value, is_multiple):
        """Build the editor widget for a field, in its shared or mixed state."""
        editor = None

        if field.type == 'string':
            editor = QLineEdit()
            if field.max_length:
                editor.setMaxLength(field.max_length)
            editor.setText("" if is_multiple else str(value))
            editor.textEdited.connect(lambda _text, e=editor, f=field: self._on_edited(e, f))
            editor.editingFinished.connect(lambda f=field: self._commit(f))

        elif field.type == 'text':
            editor = QPlainTextEdit()
            editor.setFixedHeight(app_theme.scale_int(54))
            editor.setPlainText("" if is_multiple else str(value))
            # Connected after the initial fill, so it only fires for real edits.
            editor.textChanged.connect(lambda e=editor, f=field: self._on_edited(e, f))
            # QPlainTextEdit has no editingFinished; also commit when focus leaves.
            editor.focusOutEvent = self._wrap_focus_out(editor, field)

        elif field.type == 'bool':
            editor = QCheckBox()
            editor.setTristate(is_multiple)
            if is_multiple:
                editor.setCheckState(Qt.PartiallyChecked)
            else:
                editor.setChecked(bool(value))
            editor.clicked.connect(lambda _checked, e=editor, f=field: self._on_edited(e, f, now=True))

        elif field.type == 'int':
            editor = QSpinBox()
            editor.setRange(int(field.minimum) if field.minimum is not None else -2147483648,
                            int(field.maximum) if field.maximum is not None else 2147483647)
            if field.step:
                editor.setSingleStep(int(field.step))
            if is_multiple:
                editor.setSpecialValueText(MULTIPLE_TEXT)
                editor.setValue(editor.minimum())
            else:
                editor.setValue(int(value))
            # Connected after the initial fill, so it only fires for real edits.
            editor.valueChanged.connect(lambda _v, e=editor, f=field: self._on_edited(e, f))
            editor.editingFinished.connect(lambda f=field: self._commit(f))

        elif field.type == 'float':
            editor = QDoubleSpinBox()
            editor.setDecimals(int(field.decimals))
            editor.setRange(float(field.minimum) if field.minimum is not None else -1e12,
                            float(field.maximum) if field.maximum is not None else 1e12)
            if field.step:
                editor.setSingleStep(float(field.step))
            if is_multiple:
                editor.setSpecialValueText(MULTIPLE_TEXT)
                editor.setValue(editor.minimum())
            else:
                editor.setValue(float(value))
            # Connected after the initial fill, so it only fires for real edits.
            editor.valueChanged.connect(lambda _v, e=editor, f=field: self._on_edited(e, f))
            editor.editingFinished.connect(lambda f=field: self._commit(f))

        elif field.type == 'choice':
            editor = QComboBox()
            editor.addItems(field.options)
            if is_multiple:
                # A -1 index renders blank and is never written back.
                editor.insertItem(0, MULTIPLE_TEXT)
                editor.setCurrentIndex(0)
            else:
                index = editor.findText(str(value))
                editor.setCurrentIndex(index if index >= 0 else -1)
            editor.activated.connect(lambda _index, e=editor, f=field: self._on_edited(e, f, now=True))

        elif field.type == 'multichoice':
            editor = QListWidget()
            editor.setFixedHeight(app_theme.scale_int(72))
            selected = [] if is_multiple else list(value or [])
            for option in field.options:
                list_item = QListWidgetItem(option)
                list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
                list_item.setCheckState(Qt.Checked if option in selected else Qt.Unchecked)
                editor.addItem(list_item)
            if is_multiple:
                editor.setToolTip(MULTIPLE_TEXT)
            # Connected after population, so it only fires for real edits.
            editor.itemChanged.connect(lambda _item, e=editor, f=field: self._on_edited(e, f, now=True))

        elif field.type == 'date':
            editor = QDateEdit()
            editor.setCalendarPopup(True)
            editor.setDisplayFormat("yyyy-MM-dd")
            editor.setSpecialValueText(MULTIPLE_TEXT if is_multiple else "")
            date = QDate.fromString(str(value), "yyyy-MM-dd") if value else QDate()
            editor.setDate(date if date.isValid() and not is_multiple else editor.minimumDate())
            # Connected after the initial fill, so it only fires for real edits.
            editor.dateChanged.connect(lambda _d, e=editor, f=field: self._on_edited(e, f, now=True))

        if editor is None:
            editor = QLineEdit(str(value))
            editor.setReadOnly(True)

        # Remembered so _commit can tell "user never touched this" from
        # "user deliberately chose the blank/minimum value".
        editor.setProperty("is_multiple", is_multiple)
        if field.description:
            editor.setToolTip(field.description)
        return editor

    def _mark_touched(self, editor):
        """Record that the user actually edited a mixed-state widget.

        Text and checklist editors render a mixed selection as simply empty,
        which is indistinguishable from a deliberate "clear this". Until the
        user types or ticks something, the widget stays flagged as a
        placeholder and _read_editor refuses to write it back -- otherwise
        merely selecting several annotations and clicking away would blank the
        field on every one of them.
        """
        editor.setProperty("is_multiple", False)

    def _on_edited(self, editor, field, now=False):
        """Handle a live change to an editor.

        Edits are saved as they are made rather than on focus-out, because the
        user can leave an annotation by hotkey without the editor ever losing
        focus -- which used to discard whatever they had just typed.

        Typing is debounced so a held key does not trigger one write per
        keystroke across a large selection; discrete choices (a checkbox, a
        dropdown, a date) commit immediately.
        """
        if self._populating:
            return

        self._mark_touched(editor)

        if now:
            self._commit(field)
            return

        # A different field was mid-edit; write it before starting a new timer.
        if self._pending_field is not None and self._pending_field is not field:
            self.flush_pending()

        self._pending_field = field
        self._commit_timer.start()

    def flush_pending(self):
        """Write any debounced edit immediately.

        Called before the selection changes and when the panel is hidden, so a
        pending edit is never lost to a hotkey or a dock being closed.
        """
        self._commit_timer.stop()
        field = self._pending_field
        self._pending_field = None
        if field is not None:
            self._commit(field)

    def hideEvent(self, event):
        """Flush a pending edit when the dock is hidden."""
        self.flush_pending()
        super().hideEvent(event)

    def _wrap_focus_out(self, editor, field):
        """Return a focusOutEvent that commits a QPlainTextEdit's contents."""
        original = type(editor).focusOutEvent

        def handler(event):
            original(editor, event)
            self._commit(field)

        return handler

    def _read_editor(self, field, editor):
        """Read a value out of an editor. Returns (ok, value).

        ok is False when the widget is still showing the mixed-selection
        placeholder, which must never be written back.
        """
        was_multiple = bool(editor.property("is_multiple"))

        if field.type == 'string':
            if was_multiple:
                return False, None
            return True, editor.text()

        if field.type == 'text':
            if was_multiple:
                return False, None
            return True, editor.toPlainText()

        if field.type == 'bool':
            if editor.checkState() == Qt.PartiallyChecked:
                return False, None
            return True, editor.isChecked()

        if field.type in ('int', 'float'):
            if was_multiple and editor.value() == editor.minimum():
                return False, None
            return True, editor.value()

        if field.type == 'choice':
            text = editor.currentText()
            if text == MULTIPLE_TEXT or editor.currentIndex() < 0:
                return False, None
            return True, text

        if field.type == 'multichoice':
            if was_multiple:
                return False, None
            values = []
            for index in range(editor.count()):
                item = editor.item(index)
                if item.checkState() == Qt.Checked:
                    values.append(item.text())
            return True, values

        if field.type == 'date':
            if was_multiple and editor.date() == editor.minimumDate():
                return False, None
            return True, editor.date().toString("yyyy-MM-dd")

        return False, None

    def _commit(self, field):
        """Write an editor's value to every selected annotation."""
        if self._pending_field is field:
            self._commit_timer.stop()
            self._pending_field = None

        if self._populating or not self.annotations:
            return

        editor = self._editors.get(field.name)
        if editor is None:
            return

        ok, value = self._read_editor(field, editor)
        if not ok:
            return  # Still showing the mixed-selection placeholder.

        try:
            coerced = field.coerce(value)
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Invalid Value",
                                f"'{value}' is not valid for field '{field.label}':\n{e}")
            self.rebuild()
            return

        count = len(self.annotations)
        show_progress = count > PROGRESS_THRESHOLD
        progress_bar = None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if show_progress:
                progress_bar = ProgressBar(self, title="Updating Metadata")
                progress_bar.show()
                progress_bar.start_progress(count)

            changed = 0
            for annotation in self.annotations:
                if self.schema.set_value(annotation, field.name, coerced):
                    changed += 1
                if progress_bar:
                    progress_bar.update_progress()

        finally:
            # Restored here so an exception cannot strand the app under a wait
            # cursor with an orphaned progress bar.
            if progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()
            QApplication.restoreOverrideCursor()

        if changed:
            noun = "annotation" if changed == 1 else "annotations"
            self.show_status(f"Updated {field.label} on {changed} {noun}.", 5000)

        # Deliberately no rebuild here: this runs while the user is still
        # typing, and rebuilding would destroy the widget under the cursor.
        # The next selection change refreshes the editors instead.

    # ------------------------------------------------------------------
    # Field CRUD
    # ------------------------------------------------------------------

    def get_selected_field_name(self):
        """Return the schema field name selected in the tree, or None."""
        items = self.tree.selectedItems()
        if not items:
            return None
        return items[0].data(0, Qt.UserRole)

    def select_field(self, name):
        """Select the tree row for a schema field, if it is present."""
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                if child.data(0, Qt.UserRole) == name:
                    self.tree.setCurrentItem(child)
                    return True
        return False

    def update_button_states(self):
        """Enable the field actions only when a schema field is selected."""
        name = self.get_selected_field_name()
        has_field = name is not None
        self.edit_field_button.setEnabled(has_field)
        self.delete_field_button.setEnabled(has_field)


    def _prepare_for_dialog(self):
        """Save pending edits and clear the annotation selection.

        Backspace and Delete are wired to "delete the selected annotation" in
        QtEventFilter, so leaving an annotation selected while the user types
        into a field dialog risks destroying it. Unselecting first mirrors what
        AddLabelDialog does for the same reason.
        """
        self.flush_pending()
        try:
            self.annotation_window.unselect_annotations()
        except Exception as e:
            print(f"Error clearing selection before dialog: {e}")

    def open_add_field_dialog(self):
        """Add a new metadata field to the project schema."""
        self._prepare_for_dialog()
        dialog = AddFieldDialog(self.schema, self)
        if dialog.exec_():
            field = dialog.get_field()
            self.schema.add_field(field)
            self.rebuild()
            self.select_field(field.name)
            self.show_status(f"Added field {field.label}.", 3000)
            self.schemaChanged.emit()

    def open_edit_field_dialog(self):
        """Edit the selected field, confirming any lossy consequence first."""
        name = self.get_selected_field_name()
        field = self.schema.get_field(name) if name else None
        if field is None:
            return

        self._prepare_for_dialog()
        dialog = EditFieldDialog(self.schema, field, self)
        if not dialog.exec_():
            return

        updated = dialog.get_field()
        annotations = self.get_all_annotations()

        # Only interrupt the user when the change actually costs them data.
        if not confirm_edit_field(self, self.schema, field, updated, annotations):
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            migrated = 0
            if updated.name != field.name:
                migrated = self.schema.rename(annotations, field.name, updated.name)

            # Swap the definition in place so display order is preserved.
            self.schema.fields[self.schema.fields.index(field)] = updated
            _kept, discarded = self.schema.recoerce(annotations, updated)
        finally:
            QApplication.restoreOverrideCursor()

        if migrated:
            self.show_status(f"Migrated {field.name} -> {updated.name} on {migrated} annotations.", 5000)
        elif discarded:
            self.show_status(f"Updated {updated.label}; {discarded} values discarded.", 5000)
        else:
            self.show_status(f"Updated field {updated.label}.", 3000)

        self.rebuild()
        self.select_field(updated.name)
        self.schemaChanged.emit()

    def delete_selected_field(self):
        """Delete the selected field and its stored values across the project."""
        name = self.get_selected_field_name()
        field = self.schema.get_field(name) if name else None
        if field is None:
            return

        self._prepare_for_dialog()

        annotations = self.get_all_annotations()
        if not confirm_delete_field(self, self.schema, field, annotations):
            return

        count = len(annotations)
        show_progress = count > PROGRESS_THRESHOLD
        progress_bar = None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if show_progress:
                progress_bar = ProgressBar(self, title="Removing Metadata Field")
                progress_bar.show()
                progress_bar.start_progress(count)
                removed = 0
                for annotation in annotations:
                    stored = getattr(annotation, 'metadata', None)
                    if stored and field.name in stored:
                        del stored[field.name]
                        removed += 1
                    progress_bar.update_progress()
            else:
                removed = self.schema.prune(annotations, field.name)

            self.schema.remove_field(field.name)
        finally:
            if progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()
            QApplication.restoreOverrideCursor()

        self.show_status(f"Removed field {field.label} from {removed} annotations.", 5000)
        self.rebuild()
        self.schemaChanged.emit()

    def open_visibility_dialog(self):
        """Choose which fields the grid shows, and in what order."""
        self._prepare_for_dialog()

        dialog = FieldVisibilityDialog(self.schema, self)
        if dialog.exec_():
            dialog.apply()
            self.rebuild()
            self.schemaChanged.emit()

    def adopt_from_data(self):
        """Promote scalar keys from annotation.data into real schema fields."""
        annotations = self.get_all_annotations()
        if not annotations:
            QMessageBox.information(self, "No Annotations",
                                    "There are no annotations to adopt metadata from.")
            return

        count = len(annotations)
        show_progress = count > 500
        progress_bar = None

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if show_progress:
                progress_bar = ProgressBar(self, title="Adopting Metadata Fields")
                progress_bar.show()
                progress_bar.start_progress(1)

            added, promoted, dropped = self.schema.promote_from_data(annotations)

            if progress_bar:
                progress_bar.update_progress()
        finally:
            if progress_bar:
                progress_bar.stop_progress()
                progress_bar.close()
            QApplication.restoreOverrideCursor()

        if added or promoted or dropped:
            self.show_status(f"Adopted {len(added)} fields from imported data.", 5000)
            self.rebuild()
            self.schemaChanged.emit()

        report_promotion(self, added, promoted, dropped)

    def promote_imported(self, annotations=None):
        """Promote imported data into schema fields without interrupting the user.

        Called by the importers once their annotations are in the store. Unlike
        the toolbar's Adopt action this reports only on the status bar, since it
        runs on the back of an import that already has its own dialog.
        """
        try:
            annotations = list(annotations) if annotations is not None else self.get_all_annotations()
            if not annotations:
                return

            added, promoted, _dropped = self.schema.promote_from_data(annotations)
            if added or promoted:
                self.rebuild()
                self.schemaChanged.emit()
                self.show_status(
                    f"Adopted {len(added)} metadata field(s) from imported data.", 5000
                )
        except Exception as e:
            # Never let metadata adoption take down an otherwise good import.
            print(f"Error promoting imported metadata: {e}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def show_status(self, message, timeout=5000):
        """Post a message to the main window status bar, if there is one."""
        try:
            self.main_window.status_bar.showMessage(message, timeout)
        except Exception:
            pass
