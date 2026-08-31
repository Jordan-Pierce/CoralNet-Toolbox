import warnings

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                             QLineEdit, QComboBox, QPushButton, QMessageBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QPlainTextEdit, QLabel, QWidget)

from coralnet_toolbox.MetaData.QtMetaDataSchema import FIELD_TYPES
from coralnet_toolbox.MetaData.QtMetaDataSchema import MetaDataField

from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


TYPE_LABELS = {
    'string': 'Short Text',
    'text': 'Long Text',
    'bool': 'Yes / No',
    'int': 'Whole Number',
    'float': 'Decimal Number',
    'choice': 'Dropdown (one)',
    'multichoice': 'Checklist (many)',
    'date': 'Date',
}


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FieldDialogBase(QDialog):
    """Shared form for defining a metadata field.

    The type selector drives which option rows are shown, so the user only ever
    sees the settings that apply to the type they picked.
    """

    def __init__(self, schema, field=None, parent=None):
        """Initialize the dialog, optionally seeded from an existing field."""
        super().__init__(parent)
        self.schema = schema
        self.original_field = field

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.resize(460, 480)

        self.layout = QVBoxLayout(self)

        self.setup_info_group()
        self.setup_basics_group()
        self.setup_options_group()
        self.setup_range_group()
        self.setup_default_group()
        self.setup_buttons()

        if field is not None:
            self.load_field(field)

        self.on_type_changed()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def info_text(self):
        """Return the explanatory text for this dialog. Overridden per action."""
        return ""

    def setup_info_group(self):
        """Informational header explaining what this dialog does."""
        group_box = QGroupBox("About")
        layout = QVBoxLayout()

        info_label = QLabel(self.info_text())
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_basics_group(self):
        """Set up the name / type / description rows."""
        group_box = QGroupBox("Field")
        layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. bleaching_pct")
        self.name_input.setToolTip("Unique key for this field.\nUsed as the column name in exports.")
        layout.addRow("Name:", self.name_input)

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Defaults to the name")
        self.label_input.setToolTip("Display name shown in the Metadata panel.")
        layout.addRow("Display Name:", self.label_input)

        self.type_combo = QComboBox()
        for type_name in FIELD_TYPES:
            self.type_combo.addItem(TYPE_LABELS[type_name], type_name)
        self.type_combo.setToolTip("Determines which editor the panel shows for this field.")
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        layout.addRow("Type:", self.type_combo)

        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("Optional")
        self.description_input.setToolTip("Shown as a tooltip in the Metadata panel.")
        layout.addRow("Description:", self.description_input)

        group_box.setLayout(layout)
        self.layout.addWidget(group_box)

    def setup_options_group(self):
        """Set up the dropdown/checklist option editor."""
        self.options_group = QGroupBox("Options")
        layout = QVBoxLayout()

        hint = QLabel("One option per line.")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        self.options_input = QPlainTextEdit()
        self.options_input.setFixedHeight(90)
        self.options_input.setToolTip("The choices offered by this field, one per line.")
        self.options_input.textChanged.connect(self.refresh_default_options)
        layout.addWidget(self.options_input)

        self.options_group.setLayout(layout)
        self.layout.addWidget(self.options_group)

    def setup_range_group(self):
        """Set up the numeric range rows."""
        self.range_group = QGroupBox("Range")
        layout = QFormLayout()

        self.min_check = QCheckBox("Minimum")
        self.min_input = QDoubleSpinBox()
        self.min_input.setDecimals(6)
        self.min_input.setRange(-1e12, 1e12)
        self.min_input.setEnabled(False)
        self.min_check.toggled.connect(self.min_input.setEnabled)
        min_row = QHBoxLayout()
        min_row.addWidget(self.min_check)
        min_row.addWidget(self.min_input)
        min_widget = QWidget()
        min_widget.setLayout(min_row)
        layout.addRow(min_widget)

        self.max_check = QCheckBox("Maximum")
        self.max_input = QDoubleSpinBox()
        self.max_input.setDecimals(6)
        self.max_input.setRange(-1e12, 1e12)
        self.max_input.setEnabled(False)
        self.max_check.toggled.connect(self.max_input.setEnabled)
        max_row = QHBoxLayout()
        max_row.addWidget(self.max_check)
        max_row.addWidget(self.max_input)
        max_widget = QWidget()
        max_widget.setLayout(max_row)
        layout.addRow(max_widget)

        self.step_input = QDoubleSpinBox()
        self.step_input.setDecimals(6)
        self.step_input.setRange(0.0, 1e6)
        self.step_input.setValue(1.0)
        self.step_input.setToolTip("Amount the spin box moves per click. 0 leaves it at the default.")
        layout.addRow("Step:", self.step_input)

        self.decimals_input = QSpinBox()
        self.decimals_input.setRange(0, 10)
        self.decimals_input.setValue(2)
        self.decimals_input.setToolTip("Decimal places shown for a decimal number field.")
        layout.addRow("Decimals:", self.decimals_input)

        self.max_length_input = QSpinBox()
        self.max_length_input.setRange(0, 10000)
        self.max_length_input.setValue(0)
        self.max_length_input.setToolTip("Maximum characters. 0 means unlimited.")
        layout.addRow("Max Length:", self.max_length_input)

        self.range_group.setLayout(layout)
        self.layout.addWidget(self.range_group)

    def setup_default_group(self):
        """Set up the default-value editor."""
        self.default_group = QGroupBox("Default Value")
        layout = QFormLayout()

        # One widget per shape of default; only the relevant one is shown.
        self.default_text = QLineEdit()
        self.default_text.setToolTip("Value used when an annotation has not been given one.\n"
                                     "Annotations at the default store nothing, keeping projects small.")
        layout.addRow("Default:", self.default_text)

        self.default_bool = QCheckBox("Checked by default")
        layout.addRow(self.default_bool)

        self.default_choice = QComboBox()
        layout.addRow("Default Option:", self.default_choice)

        self.default_group.setLayout(layout)
        self.layout.addWidget(self.default_group)

    def setup_buttons(self):
        """Set up the OK / Cancel row."""
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.validate_and_accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)

    # ------------------------------------------------------------------
    # Dynamic form behaviour
    # ------------------------------------------------------------------

    def current_type(self):
        """Return the selected field type key."""
        return self.type_combo.currentData()

    def on_type_changed(self):
        """Show only the settings that apply to the selected type."""
        field_type = self.current_type()

        is_choice = field_type in ('choice', 'multichoice')
        is_number = field_type in ('int', 'float')
        is_string = field_type in ('string', 'text')

        self.options_group.setVisible(is_choice)
        self.range_group.setVisible(is_number or is_string)

        self.min_check.setVisible(is_number)
        self.min_input.setVisible(is_number)
        self.max_check.setVisible(is_number)
        self.max_input.setVisible(is_number)
        self.step_input.setVisible(is_number)
        self.decimals_input.setVisible(field_type == 'float')
        self.max_length_input.setVisible(is_string)

        # Default editor: a checkbox for bool, a combo for a single choice,
        # a line edit for everything else. A checklist default is left empty.
        self.default_bool.setVisible(field_type == 'bool')
        self.default_choice.setVisible(field_type == 'choice')
        self.default_text.setVisible(field_type not in ('bool', 'choice', 'multichoice'))
        self.default_group.setVisible(field_type != 'multichoice')

        if field_type == 'date':
            self.default_text.setPlaceholderText("YYYY-MM-DD, or blank")
        elif is_number:
            self.default_text.setPlaceholderText("0")
        else:
            self.default_text.setPlaceholderText("Blank")

        if is_choice:
            self.refresh_default_options()

        self.adjustSize()

    def get_options(self):
        """Parse the options box into a de-duplicated list."""
        options = []
        for line in self.options_input.toPlainText().splitlines():
            text = line.strip()
            if text and text not in options:
                options.append(text)
        return options

    def refresh_default_options(self):
        """Keep the default-option combo in step with the options list."""
        previous = self.default_choice.currentText()
        self.default_choice.clear()
        self.default_choice.addItems(self.get_options())
        index = self.default_choice.findText(previous)
        if index >= 0:
            self.default_choice.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Load / build
    # ------------------------------------------------------------------

    def load_field(self, field):
        """Populate the form from an existing field definition."""
        self.name_input.setText(field.name)
        self.label_input.setText(field.label)
        self.description_input.setText(field.description)

        index = self.type_combo.findData(field.type)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

        if field.options:
            self.options_input.setPlainText("\n".join(field.options))

        if field.minimum is not None:
            self.min_check.setChecked(True)
            self.min_input.setValue(float(field.minimum))
        if field.maximum is not None:
            self.max_check.setChecked(True)
            self.max_input.setValue(float(field.maximum))
        if field.step is not None:
            self.step_input.setValue(float(field.step))
        self.decimals_input.setValue(int(field.decimals))
        self.max_length_input.setValue(int(field.max_length or 0))

        if field.type == 'bool':
            self.default_bool.setChecked(bool(field.default))
        elif field.type == 'choice':
            self.refresh_default_options()
            index = self.default_choice.findText(str(field.default))
            if index >= 0:
                self.default_choice.setCurrentIndex(index)
        elif field.type != 'multichoice':
            self.default_text.setText("" if field.default in ('', None) else str(field.default))

    def build_field(self):
        """Construct a MetaDataField from the form. Raises ValueError if invalid."""
        field_type = self.current_type()
        options = self.get_options() if field_type in ('choice', 'multichoice') else None

        if field_type == 'bool':
            default = self.default_bool.isChecked()
        elif field_type == 'choice':
            default = self.default_choice.currentText() or None
        elif field_type == 'multichoice':
            default = []
        else:
            text = self.default_text.text().strip()
            default = text if text else None

        is_number = field_type in ('int', 'float')
        is_string = field_type in ('string', 'text')

        return MetaDataField(
            name=self.name_input.text().strip(),
            type=field_type,
            label=self.label_input.text().strip() or None,
            default=default,
            options=options,
            minimum=self.min_input.value() if (is_number and self.min_check.isChecked()) else None,
            maximum=self.max_input.value() if (is_number and self.max_check.isChecked()) else None,
            step=self.step_input.value() if (is_number and self.step_input.value() > 0) else None,
            decimals=self.decimals_input.value(),
            max_length=self.max_length_input.value() if is_string else 0,
            description=self.description_input.text().strip(),
            visible=self.original_field.visible if self.original_field else True,
            taglab_key=self.original_field.taglab_key if self.original_field else None,
        )

    def get_field(self):
        """Return the field built by the last successful validation."""
        return self._field

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_and_accept(self):
        """Validate the form and accept the dialog if it holds together."""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "A field name is required.")
            return

        existing = self.schema.get_field(name)
        if existing is not None and existing is not self.original_field:
            QMessageBox.warning(self, "Field Exists",
                                f"A metadata field named '{name}' already exists.")
            return

        field_type = self.current_type()
        if field_type in ('choice', 'multichoice') and not self.get_options():
            QMessageBox.warning(self, "Input Error",
                                "A dropdown or checklist needs at least one option.")
            return

        if field_type in ('int', 'float') and self.min_check.isChecked() and self.max_check.isChecked():
            if self.min_input.value() > self.max_input.value():
                QMessageBox.warning(self, "Input Error",
                                    "The minimum cannot be greater than the maximum.")
                return

        try:
            self._field = self.build_field()
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "Invalid Field", str(e))
            return

        self.accept()


class AddFieldDialog(FieldDialogBase):
    """Dialog for defining a new metadata field."""

    def __init__(self, schema, parent=None):
        """Initialize the Add Field dialog."""
        super().__init__(schema, field=None, parent=parent)
        self.setWindowTitle("Add Metadata Field")
        self.setObjectName("AddFieldDialog")

    def info_text(self):
        """Explain what adding a field does."""
        return (
            "Define a new metadata field for <b>every annotation in this project</b>. "
            "The type you choose determines the editor shown in the Metadata panel."
            "<br><br>"
            "Only values that differ from the <b>default</b> are stored, so a field costs "
            "nothing on annotations you never fill in -- choose the default accordingly."
            "<br><br>"
            "The <b>name</b> is the column name used when exporting, and must be unique."
        )


class EditFieldDialog(FieldDialogBase):
    """Dialog for changing an existing metadata field."""

    def __init__(self, schema, field, parent=None):
        """Initialize the Edit Field dialog for the given field."""
        super().__init__(schema, field=field, parent=parent)
        self.setWindowTitle(f"Edit Metadata Field - {field.label}")
        self.setObjectName("EditFieldDialog")

    def info_text(self):
        """Explain what editing a field can cost."""
        return (
            "Change this field's definition across <b>the whole project</b>."
            "<br><br>"
            "Renaming it, changing its type, narrowing its range, or removing an option "
            "can affect values already stored on your annotations. You will be shown "
            "exactly what is at stake before anything is applied."
            "<br><br>"
            "Changing only the display name or description is always safe."
        )
