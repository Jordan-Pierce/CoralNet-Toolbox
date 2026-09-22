import os
import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from coralnet_toolbox.Morphological.overlap_ops import MERGE, REMOVE, SUBTRACT, OverlapSpec

from coralnet_toolbox.QtProgressBar import ProgressBar

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

# (key, combo text, description, count columns after "Image")
# Descriptions are kept about the same length so switching operations doesn't resize the dialog.
OPERATIONS = [
    (
        SUBTRACT,
        "Subtract overlap",
        "Cut the overlapping labels out of the labels to change, leaving holes where needed. "
        "Overlapping annotations stay as they are. Polygons only.",
        ["Checked", "Clipped", "Removed"],
    ),
    (
        REMOVE,
        "Remove overlapping",
        "Delete any annotation of the labels to change that overlaps the overlapping labels. "
        "Overlapping annotations stay as they are. Polygons only.",
        ["Checked", "Removed"],
    ),
    (
        MERGE,
        "Merge same label",
        "Combine annotations of the labels to merge that overlap each other into one shape. "
        "Only annotations sharing a label are merged. Polygons only.",
        ["Checked", "Groups", "Merged"],
    ),
]


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class LabelChecklist(QGroupBox):
    """Checkable list of project labels with a colour swatch and All / None buttons."""

    def __init__(self, title, on_change, parent=None):
        super().__init__(title, parent)
        self.on_change = on_change

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(140)
        self.list_widget.itemChanged.connect(lambda _item: self.on_change())
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self.set_all(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self.set_all(False))
        button_layout.addWidget(all_button)
        button_layout.addWidget(none_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def populate(self, labels):
        """Refill with labels, keeping whatever was checked before."""
        checked = self.checked_ids()

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for label in labels:
            item = QListWidgetItem(f"{label.short_label_code} - {label.long_label_code}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if label.id in checked else Qt.Unchecked)
            item.setData(Qt.UserRole, label.id)

            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(label.color))
            item.setIcon(QIcon(pixmap))

            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def set_all(self, checked):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.list_widget.blockSignals(False)
        self.on_change()

    def checked_ids(self):
        ids = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                ids.add(item.data(Qt.UserRole))
        return ids


class OverlapOperationsTab(QWidget):
    """Subtract, remove or merge overlapping vector annotations by label, across the
    images highlighted in the ImageWindow.

    Preview works out the changes without making them and selects the affected
    annotations on the current image. Apply works them out again against the
    current state and makes them as a single undoable step.

    The Preview and Apply buttons live in ``action_widget``, which the dialog puts
    in its shared bottom row rather than in this tab.
    """

    # There is no options groupbox yet; these are the values it would control.
    MIN_OVERLAP = 0.0           # any overlap counts
    MIN_PIECE_AREA = 50.0       # px^2; slivers left by a subtract along shared edges are dropped
    INCLUDE_UNVERIFIED = True

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.annotation_window = dialog.annotation_window
        self.main_window = dialog.main_window
        self.image_window = dialog.image_window

        self.layout = QVBoxLayout(self)

        self.setup_operation_layout()
        self.setup_labels_layout()
        self.setup_results_layout()
        self.setup_action_widget()

        self.on_operation_changed()

    # ------------------------------------------------------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------------------------------------------------------

    def setup_operation_layout(self):
        """Operation picker and its description."""
        group_box = QGroupBox("Operation")
        layout = QVBoxLayout(group_box)

        self.operation_combo = QComboBox()
        for key, text, _description, _columns in OPERATIONS:
            self.operation_combo.addItem(text, key)
        self.operation_combo.currentIndexChanged.connect(self.on_operation_changed)
        layout.addWidget(self.operation_combo)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        self.layout.addWidget(group_box)

    def setup_labels_layout(self):
        """Side by side label lists: what changes, and what it is tested against."""
        layout = QHBoxLayout()

        self.target_list = LabelChecklist("Labels to change", self.clear_results)
        self.target_list.setToolTip("Annotations with these labels are the ones modified or deleted.")
        layout.addWidget(self.target_list)

        self.reference_list = LabelChecklist("Overlapping labels", self.clear_results)
        self.reference_list.setToolTip("Annotations with these labels are used for the test and are never modified.")
        layout.addWidget(self.reference_list)

        self.layout.addLayout(layout)

    def setup_results_layout(self):
        """Per image counts from the last preview or apply."""
        group_box = QGroupBox("Results")
        layout = QVBoxLayout(group_box)

        self.results_caption = QLabel("Press Preview to see what would change.")
        self.results_caption.setWordWrap(True)
        layout.addWidget(self.results_caption)

        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionMode(QTableWidget.NoSelection)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setMinimumHeight(150)
        self.results_table.setMaximumHeight(240)
        layout.addWidget(self.results_table)

        self.layout.addWidget(group_box)

    def setup_action_widget(self):
        """Preview and Apply, in a widget the dialog places in its bottom row."""
        self.action_widget = QWidget()
        button_layout = QHBoxLayout(self.action_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip("Work out the changes without making them, and select the "
                                       "affected annotations on the current image.")
        self.preview_button.clicked.connect(self.preview)
        button_layout.addWidget(self.preview_button)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Make the changes on the highlighted images. Undo with Ctrl+Z.")
        self.apply_button.clicked.connect(self.apply)
        button_layout.addWidget(self.apply_button)

    # ------------------------------------------------------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------------------------------------------------------

    def current_operation(self):
        """Return (key, text, description, columns) for the selected operation."""
        return OPERATIONS[max(self.operation_combo.currentIndex(), 0)]

    def refresh_labels(self):
        """Repopulate both label lists from the project, keeping checks."""
        labels = list(self.main_window.label_window.labels)
        self.target_list.populate(labels)
        self.reference_list.populate(labels)

    def on_operation_changed(self):
        """Show only the controls the selected operation uses."""
        key, _text, description, _columns = self.current_operation()
        self.description_label.setText(description)

        uses_reference = key in (SUBTRACT, REMOVE)
        self.reference_list.setVisible(uses_reference)
        self.target_list.setTitle("Labels to change" if uses_reference else "Labels to merge")

        self.clear_results()

    def clear_results(self):
        """Drop results that no longer match the settings."""
        _key, _text, _description, columns = self.current_operation()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(len(columns) + 1)
        self.results_table.setHorizontalHeaderLabels(["Image"] + columns)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(columns) + 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.results_caption.setText("Press Preview to see what would change.")

    def build_spec(self):
        """Read the controls into an OverlapSpec, or explain what is missing and return None."""
        key = self.current_operation()[0]
        target_ids = self.target_list.checked_ids()
        reference_ids = self.reference_list.checked_ids() if key != MERGE else set()

        if not target_ids:
            noun = "merge" if key == MERGE else "change"
            QMessageBox.warning(self, "No Labels", f"Check at least one label to {noun}.")
            return None

        if key != MERGE:
            if not reference_ids:
                QMessageBox.warning(self, "No Labels", "Check at least one overlapping label.")
                return None
            if target_ids & reference_ids:
                QMessageBox.warning(
                    self, "Same Label On Both Sides",
                    "A label cannot be both a label to change and an overlapping label. "
                    "Uncheck it in one of the lists."
                )
                return None

        return OverlapSpec(
            operation=key,
            target_label_ids=frozenset(target_ids),
            reference_label_ids=frozenset(reference_ids),
            min_overlap=self.MIN_OVERLAP,
            min_piece_area=self.MIN_PIECE_AREA,
            include_unverified=self.INCLUDE_UNVERIFIED,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------------------------------------------------------

    def collect_plans(self, image_paths, spec, title):
        """Plan spec on every image, with a progress bar when there are several."""
        aw = self.annotation_window

        progress_bar = None
        if len(image_paths) > 1:
            progress_bar = ProgressBar(self, title=title)
            progress_bar.show()
            progress_bar.start_progress(len(image_paths))

        plans = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for image_path in image_paths:
                plans.append(aw.plan_overlap_operation(image_path, spec))
                if progress_bar is not None:
                    progress_bar.update_progress()
        finally:
            QApplication.restoreOverrideCursor()
            if progress_bar is not None:
                progress_bar.stop_progress()
                progress_bar.close()
        return plans

    def highlighted_paths(self):
        """Highlighted image paths, or None after telling the user to highlight some."""
        image_paths = self.image_window.table_model.get_highlighted_paths()
        if not image_paths:
            QMessageBox.warning(self, "No Selection", "Please highlight at least one image row.")
            return None
        return image_paths

    def set_controls_enabled(self, enabled):
        self.preview_button.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)

    def preview(self):
        """Work out the changes, show the counts, and select what would change on the current image."""
        spec = self.build_spec()
        if spec is None:
            return
        image_paths = self.highlighted_paths()
        if image_paths is None:
            return

        self.set_controls_enabled(False)
        try:
            plans = self.collect_plans(image_paths, spec, "Previewing Overlaps")
        finally:
            self.set_controls_enabled(True)

        self.show_results(plans, applied=False)

        aw = self.annotation_window
        current = [a for plan in plans if plan.image_path == aw.current_image_path for a in plan.affected]
        aw.unselect_annotations()
        if current:
            aw.select_annotations_bulk(current)

    def apply(self):
        """Work out the changes against the current state and make them as one undo step."""
        spec = self.build_spec()
        if spec is None:
            return
        image_paths = self.highlighted_paths()
        if image_paths is None:
            return

        self.set_controls_enabled(False)
        try:
            plans = self.collect_plans(image_paths, spec, "Checking Overlaps")
            changed = [plan for plan in plans if plan.has_changes]

            if not changed:
                self.show_results(plans, applied=False)
                QMessageBox.information(self, "Nothing To Do",
                                        "None of the highlighted images have annotations to change.")
                return

            if len(changed) > 1:
                total = sum(len(plan.affected) for plan in changed)
                reply = QMessageBox.question(
                    self, "Apply To Several Images",
                    f"Change {total} annotations across {len(changed)} images?\n\n"
                    "This can be undone with Ctrl+Z.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if reply != QMessageBox.Yes:
                    return

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                result = self.annotation_window.apply_overlap_plans(changed)
            finally:
                QApplication.restoreOverrideCursor()
        finally:
            self.set_controls_enabled(True)

        self.show_results(plans, applied=True)

        if result:
            removed, added = result
            message = (f"{self.current_operation()[1]}: removed {len(removed)} and added "
                       f"{len(added)} annotations across {len(changed)} image(s).")
            try:
                self.main_window.status_bar.showMessage(message, 5000)
            except Exception:
                pass

    # ------------------------------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------------------------------

    def plan_counts(self, plan):
        """Counts for the table columns after "Image", matching OPERATIONS."""
        key = self.current_operation()[0]
        if key == SUBTRACT:
            return [plan.checked, len(plan.replaced), len(plan.removed)]
        if key == REMOVE:
            return [plan.checked, len(plan.removed)]
        return [plan.checked, len(plan.replaced), sum(len(sources) for sources, _g, _t in plan.replaced)]

    def show_results(self, plans, applied):
        """Fill the table with the images that change, plus a total row."""
        self.clear_results()

        changed = [plan for plan in plans if plan.has_changes]
        verb = "changed" if applied else "would change"
        self.results_caption.setText(f"{len(changed)} of {len(plans)} image(s) {verb}.")

        if not changed:
            return

        totals = None
        rows = []
        for plan in changed:
            counts = self.plan_counts(plan)
            totals = counts if totals is None else [a + b for a, b in zip(totals, counts)]
            rows.append((os.path.basename(plan.image_path), counts, plan.image_path))
        if len(rows) > 1:
            rows.append(("Total", totals, None))

        self.results_table.setRowCount(len(rows))
        for row, (name, counts, path) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            if path:
                name_item.setToolTip(path)
            else:
                font = name_item.font()
                font.setBold(True)
                name_item.setFont(font)
            self.results_table.setItem(row, 0, name_item)

            for column, value in enumerate(counts, start=1):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.results_table.setItem(row, column, item)
