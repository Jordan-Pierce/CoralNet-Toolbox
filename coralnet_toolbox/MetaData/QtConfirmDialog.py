import warnings

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                             QPushButton, QMessageBox)

from coralnet_toolbox.Icons import get_window_icon

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FieldConfirmDialog(QDialog):
    """Confirmation for a schema change, with the consequences spelled out.

    A plain message box has nowhere to put a standing explanation of what the
    action means, so this carries an About groupbox alongside the specific,
    count-aware consequences of the change being confirmed.
    """

    def __init__(self, title, heading, info, consequences, confirm_text="OK", parent=None):
        """Initialize the dialog.

        Args:
            title: Window title.
            heading: One-line statement of what is about to happen.
            info: Standing explanation shown in the About groupbox.
            consequences: List of HTML fragments, one per consequence.
            confirm_text: Label for the confirming button.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setWindowIcon(get_window_icon("coralnet.svg"))
        self.setWindowTitle(title)
        self.setObjectName("FieldConfirmDialog")
        self.resize(480, 320)

        self.layout = QVBoxLayout(self)

        heading_label = QLabel(f"<b>{heading}</b>")
        heading_label.setWordWrap(True)
        self.layout.addWidget(heading_label)

        info_group = QGroupBox("About")
        info_layout = QVBoxLayout()
        info_label = QLabel(info)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        self.layout.addWidget(info_group)

        consequences_group = QGroupBox("Consequences")
        consequences_layout = QVBoxLayout()
        consequences_label = QLabel(f"<ul>{''.join(consequences)}</ul>")
        consequences_label.setWordWrap(True)
        consequences_layout.addWidget(consequences_label)
        consequences_group.setLayout(consequences_layout)
        self.layout.addWidget(consequences_group)

        self.layout.addStretch()

        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton(confirm_text)
        self.confirm_button.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        # Cancel is the safe answer, so it takes Enter and initial focus.
        self.cancel_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(button_layout)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


DELETE_INFO = (
    "Deleting a metadata field removes it from the project schema and discards every "
    "value stored under it.<br><br>"
    "If you only want it out of the way, <b>hide</b> it instead using Show / Hide "
    "Fields -- hidden fields keep their values and are still saved with the project."
)

EDIT_INFO = (
    "Values are stored per annotation under this field's name, and only when they "
    "differ from its default.<br><br>"
    "That means renaming, retyping, narrowing a range, or changing the default all "
    "change how existing values are read back. The consequences below were computed "
    "against the annotations in this project."
)


def _plural(count, noun="annotation"):
    """Return 'N noun' with the noun pluralized when appropriate."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def confirm_delete_field(parent, schema, field, annotations):
    """Ask before deleting a field, stating exactly what will be lost.

    Returns True if the user confirmed.
    """
    stored = schema.count_stored(annotations, field.name)

    consequences = ["<li>The field will be removed from the project schema.</li>"]
    if stored:
        consequences.append(
            f"<li>Stored values will be <b>permanently deleted from {_plural(stored)}</b>.</li>"
        )
    else:
        consequences.append("<li>No annotation currently stores a value for it.</li>")
    consequences.append("<li>Annotations already exported are unaffected.</li>")
    consequences.append("<li>This cannot be undone.</li>")

    dialog = FieldConfirmDialog(
        title="Confirm Delete",
        heading=f"Delete metadata field '{field.label}'?",
        info=DELETE_INFO,
        consequences=consequences,
        confirm_text="Delete",
        parent=parent,
    )
    return dialog.exec_() == QDialog.Accepted


def describe_edit_consequences(schema, old_field, new_field, annotations):
    """List the ways an edit would cost the user data.

    Returns a list of HTML <li> strings; empty when the edit is purely
    cosmetic, in which case no prompt is warranted.
    """
    consequences = []
    stored = schema.count_stored(annotations, old_field.name)

    if new_field.name != old_field.name:
        consequences.append(
            f"<li>Values will migrate from '<b>{old_field.name}</b>' to "
            f"'<b>{new_field.name}</b>' on {_plural(stored)}.</li>"
        )
        consequences.append(
            "<li>Exports and scripts referencing the old column name will need updating.</li>"
        )

    if new_field.type != old_field.type:
        uncoercible = schema.count_uncoercible(annotations, new_field)
        consequences.append(
            f"<li>The type changes from <b>{old_field.type}</b> to <b>{new_field.type}</b>.</li>"
        )
        if uncoercible:
            consequences.append(
                f"<li>Values that cannot be converted will be "
                f"<b>discarded on {_plural(uncoercible)}</b>.</li>"
            )

    elif new_field.minimum != old_field.minimum or new_field.maximum != old_field.maximum:
        # A narrowed range silently clamps, which is worth saying out loud.
        clamped = 0
        for annotation in annotations:
            if not schema.has_stored_value(annotation, old_field.name):
                continue
            raw = annotation.metadata[old_field.name]
            ok, coerced = new_field.try_coerce(raw)
            if ok and coerced != raw:
                clamped += 1
        if clamped:
            consequences.append(
                f"<li>The new range will <b>clamp stored values on {_plural(clamped)}</b>.</li>"
            )

    if new_field.default != old_field.default:
        # Sparse storage means the default is not merely cosmetic: annotations
        # sitting at a default store nothing, so moving the default silently
        # changes what those annotations read back as.
        at_old_default = len(annotations) - stored
        now_compacted = sum(1 for annotation in annotations
                            if schema.has_stored_value(annotation, old_field.name)
                            and annotation.metadata[old_field.name] == new_field.default)
        consequences.append(
            f"<li>The default changes from '<b>{old_field.default}</b>' to "
            f"'<b>{new_field.default}</b>'.</li>"
        )
        if at_old_default:
            consequences.append(
                f"<li>{_plural(at_old_default)} currently sitting at the old default "
                f"will now read as '<b>{new_field.default}</b>'.</li>"
            )
        if now_compacted:
            consequences.append(
                f"<li>{_plural(now_compacted)} holding the new default explicitly "
                f"will stop storing it.</li>"
            )

    removed_options = [option for option in old_field.options if option not in new_field.options]
    if removed_options:
        affected = 0
        for annotation in annotations:
            if not schema.has_stored_value(annotation, old_field.name):
                continue
            value = annotation.metadata[old_field.name]
            values = value if isinstance(value, list) else [value]
            if any(item in removed_options for item in values):
                affected += 1
        consequences.append(
            f"<li>Option(s) <b>{', '.join(removed_options)}</b> were removed.</li>"
        )
        if affected:
            consequences.append(
                f"<li>{_plural(affected)} holding a removed option will "
                f"<b>fall back to the default</b>.</li>"
            )

    return consequences


def confirm_edit_field(parent, schema, old_field, new_field, annotations):
    """Ask before applying a lossy field edit.

    A purely cosmetic change (display name, description, tooltip, a widened
    range) is applied without interrupting the user. Returns True to proceed.
    """
    consequences = describe_edit_consequences(schema, old_field, new_field, annotations)
    if not consequences:
        return True

    dialog = FieldConfirmDialog(
        title="Confirm Field Change",
        heading=f"Apply these changes to '{old_field.label}'?",
        info=EDIT_INFO,
        consequences=consequences,
        confirm_text="Apply",
        parent=parent,
    )
    return dialog.exec_() == QDialog.Accepted


def report_promotion(parent, added, promoted, dropped):
    """Summarise what an 'Adopt from data' pass did."""
    if not added and not promoted and not dropped:
        QMessageBox.information(
            parent, "Nothing to Adopt",
            "No new metadata fields were found in the imported annotation data."
        )
        return

    parts = []
    if added:
        parts.append(f"<li>Added <b>{len(added)}</b> field(s): {', '.join(added)}</li>")
    if promoted:
        parts.append(f"<li>Moved <b>{promoted}</b> value(s) out of raw imported data.</li>")
    if dropped:
        parts.append(
            f"<li>Discarded <b>{dropped}</b> key(s) that duplicate a computed built-in "
            f"value (area, perimeter, bbox, centroid).</li>"
        )

    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setWindowTitle("Fields Adopted")
    msg_box.setText("Imported data has been adopted into the metadata schema.")
    msg_box.setInformativeText(f"<ul>{''.join(parts)}</ul>")
    msg_box.exec_()
