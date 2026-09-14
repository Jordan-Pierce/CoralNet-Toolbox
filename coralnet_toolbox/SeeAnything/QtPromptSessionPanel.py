import os
from colorsys import hsv_to_rgb

import numpy as np
import torch
from sklearn.decomposition import PCA

import pyqtgraph as pg

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPushButton, QSizePolicy, QTabWidget, QVBoxLayout, QWidget)

from coralnet_toolbox.SeeAnything.PromptSession import (KIND_TEXT, ORIGIN_ANNOTATIONS, ORIGIN_FILE,
                                                        ORIGIN_PHRASE, ORIGIN_TOOL, ORIGIN_TOOL_PHRASE,
                                                        group_rows)
from coralnet_toolbox.SeeAnything.QtPromptAlignment import PromptAlignmentWidget


# Above this many enabled negatives the panel says what they cost. Each one is a
# class the head scores on every candidate object, and each can also win an object
# the user wants. A warning rather than a cap: sampling would silently drop
# examples the user chose.
NEGATIVE_WARNING_COUNT = 50

# Where examples came from, in the order the panel lists them. The key is the
# section `section_of` puts an example in; None collects examples with no origin.
SECTION_ANNOTATIONS = ORIGIN_ANNOTATIONS
SECTION_TOOL = "tool"
SECTION_PHRASES = ORIGIN_PHRASE
SECTION_FILES = ORIGIN_FILE
SECTIONS = (
    (SECTION_ANNOTATIONS, "From annotations",
     "Added with Add highlighted in the Generator, one example per image.\n"
     "Adding an image again only changes it if its annotations changed, and these are\n"
     "embedded again automatically when the image size changes."),
    (SECTION_TOOL, "From the See Anything tool",
     "Drawn, clicked or typed (Ctrl+T) in the See Anything tool, and sent with Add from Tool.\n"
     "Their work areas are gone, so they cannot be embedded again at another image size."),
    (SECTION_PHRASES, "Phrases",
     "Text prompts added with Add phrase, or kept in Inspect's Text Alignment."),
    (SECTION_FILES, "From files",
     "Loaded from a prompt-embedding file that does not say where its examples came from."),
    (None, "Other", "Examples that do not say where they came from."),
)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def fixed_width_text(label):
    """Let a word-wrapped label take the width its layout gives it, and never ask for more.

    A wrapped QLabel's size hint grows with its text, so a label whose text changes
    (a status, a warning) widened its column every time the text got longer, and
    the other column shrank to match.
    """
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    label.setSizePolicy(policy)
    return label


def section_of(prototype):
    """The panel section an example is listed under."""
    source = prototype.source
    if source in (ORIGIN_TOOL, ORIGIN_TOOL_PHRASE):
        return SECTION_TOOL
    if source in (ORIGIN_ANNOTATIONS, ORIGIN_PHRASE, ORIGIN_FILE):
        return source
    return None


def _plural(count, word, plural=None):
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def section_counts(section, positives, negatives):
    """"3 images", "2 examples, 1 negative", "1 phrase" for one section's rows."""
    if section == SECTION_ANNOTATIONS:
        noun = "image"
    elif section == SECTION_PHRASES:
        noun = "phrase"
    else:
        noun = "example"
    parts = []
    if positives:
        parts.append(_plural(positives, noun))
    if negatives:
        parts.append(_plural(negatives, "negative"))
    return ", ".join(parts)


def describe_sources(session):
    """Where a prompt's examples came from, in one line.

    E.g. "3 images from annotations, 2 examples and 1 negative from the tool,
    1 phrase (1 switched off)".
    """
    if session.is_empty():
        return "empty"

    where = {SECTION_ANNOTATIONS: "from annotations", SECTION_TOOL: "from the tool",
             SECTION_PHRASES: "", SECTION_FILES: "from files", None: "from elsewhere"}
    parts = []
    for section, _title, _hint in SECTIONS:
        positives = sum(1 for p in session.positives if section_of(p) == section)
        negatives = sum(1 for p in session.negatives if section_of(p) == section)
        if not positives and not negatives:
            continue
        counts = section_counts(section, positives, negatives).replace(", ", " and ")
        parts.append(f"{counts} {where[section]}".strip())

    off = sum(1 for p in session.positives + session.negatives if not p.enabled)
    line = ", ".join(parts)
    if off:
        line += f" ({off} switched off)"
    return line


def row_text(prototypes, negative, imgsz=None):
    """What one panel row says; its section heading already says where it came from.

    Args:
        prototypes (list[Prototype]): The examples the row stands for.
        negative (bool): Whether they are negatives.
        imgsz (int, optional): The image size a run would use. A visual example
            embedded at another size says so, since it is what a size warning means.
    """
    sign = "-" if negative else "+"
    first = prototypes[0]
    if first.source == ORIGIN_ANNOTATIONS and len(prototypes) > 1:
        origin = first.origin
        image = os.path.basename(str(origin.get("image", "")))
        text = f"{origin.get('label', '?')}: {len(prototypes)} annotations on {image}"
    elif first.kind == KIND_TEXT:
        text = f"'{first.label}'"
        if first.source in (None, ORIGIN_TOOL_PHRASE):
            text += " (Ctrl+T)"
    else:
        text = first.label

    sizes = sorted({int(p.imgsz) for p in prototypes if p.is_visual and p.imgsz})
    if imgsz and sizes and sizes != [int(imgsz)]:
        text += f"   [embedded at {', '.join(str(s) for s in sizes)}]"
    return f"{sign}  {text}"


def row_tooltip(prototypes, negative):
    """Where a row's examples came from and what can still be done with them."""
    first = prototypes[0]
    origin = first.origin or {}
    lines = ["Find fewer like this (a negative example)." if negative else "Find more like this."]

    source = first.source
    if source == ORIGIN_ANNOTATIONS:
        lines.append(f"From the {origin.get('label', '?')} annotations on {origin.get('image', '?')}.")
        lines.append("Adding this image again only changes it if its annotations changed.")
    elif source == ORIGIN_TOOL:
        lines.append("From the See Anything tool. Its work area is gone, so it cannot be embedded again.")
    elif source == ORIGIN_TOOL_PHRASE:
        lines.append("The phrase typed in the See Anything tool with Ctrl+T.")
    elif source == ORIGIN_PHRASE:
        lines.append("A phrase added with Add phrase or kept in Inspect.")
    elif source == ORIGIN_FILE:
        lines.append("Loaded from a prompt-embedding file.")

    if first.kind == KIND_TEXT:
        lines.append("A phrase scores far higher than an example crop; the two are not comparable.")
    elif first.imgsz:
        lines.append(f"Embedded at image size {first.imgsz}.")
    if len(prototypes) > 1:
        lines.append(f"{len(prototypes)} examples that switch on and off together.")
    return "\n".join(lines)


def session_warnings(session, stem=None, imgsz=None):
    """Everything worth saying about a prompt before it runs.

    Args:
        session (PromptSession): The prompt.
        stem (str, optional): The checkpoint the prompt would run with.
        imgsz (int, optional): The image size the prompt would run at.

    Returns:
        list[str]: One sentence per problem, most serious first.
    """
    warnings = []
    if session.model_stem and stem and session.model_stem != stem:
        warnings.append(
            f"Made with '{session.model_stem}', but the model is '{stem}': these examples "
            f"cannot be used with it.")

    if session.is_mixed():
        warnings.append(
            "Mixes text and visual examples: a phrase scores far higher than an example crop "
            "of the same object, so a threshold tuned for one hides the other's detections.")

    if session.enabled_negatives() and not session.decoys_can_compete():
        warnings.append(
            "Negative examples only work against a visual positive; with text alone they are ignored.")

    negatives = len(session.enabled_negatives())
    if negatives > NEGATIVE_WARNING_COUNT:
        warnings.append(
            f"{negatives} negative examples: each is one more class scored on every object, so "
            f"runs slow a little, and each can also win objects you want. Untick images you "
            f"do not need.")

    off_size = session.off_size(imgsz)
    redo = [p for p in off_size if p.can_reembed]
    fixed = [p for p in off_size if not p.can_reembed]
    if redo:
        warnings.append(
            f"{len(redo)} example{'s were' if len(redo) != 1 else ' was'} embedded at another "
            f"image size and will be embedded again at {imgsz} when the run starts.")
    if fixed:
        sizes = ", ".join(str(s) for s in sorted({int(p.imgsz) for p in fixed}))
        warnings.append(
            f"{len(fixed)} example{'s were' if len(fixed) != 1 else ' was'} embedded at {sizes}, "
            f"not {imgsz}, and cannot be embedded again: an embedding depends on the image size, "
            f"so predict at {sizes} to use {'them' if len(fixed) != 1 else 'it'} as made.")
    return warnings


def inspect_session(parent, session, model, label_names=None, prompt_store=None):
    """Open the embedding scatter and Text Alignment for a session.

    Args:
        parent (QWidget): Parent for the dialog.
        session (PromptSession): The examples to show.
        model: The loaded YOLOE, needed to encode phrases; may be None.
        label_names (list[str], optional): Project labels to offer as phrases.
        prompt_store (optional): Where "Keep" puts a phrase -- anything with
            `add_text_prototype`, `remove_text_prototype`, `text_prototype_phrases`.

    Returns:
        bool: True if the dialog was shown.
    """
    entries = []
    for prototype in session.enabled_positives():
        entries.append((prototype.embedding.reshape(1, 1, -1), f"+ {prototype.label}"))
    for prototype in session.enabled_negatives():
        entries.append((prototype.embedding.reshape(1, 1, -1), f"- {prototype.label}"))

    # With nothing to plot the Text Alignment tab still works -- a typed phrase is
    # a prompt in its own right -- as long as there is a model to encode with.
    if not entries and model is None:
        return False

    final = None
    if entries:
        final = torch.nn.functional.normalize(
            torch.cat([vpe for vpe, _ in entries]).mean(dim=0, keepdim=True), p=2, dim=-1)

    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        dialog = VPEVisualizationDialog(entries,
                                        final,
                                        prototypes=[vpe for vpe, _ in entries],
                                        model=model,
                                        label_names=label_names,
                                        prompt_store=prompt_store,
                                        parent=parent)
    finally:
        QApplication.restoreOverrideCursor()
    dialog.exec_()
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SessionPhraseStore:
    """Lets the Text Alignment tab keep phrases as rows of a session.

    Args:
        session_source (callable): Returns the session to edit.
        encode (callable): Phrase -> embedding, from the model the session belongs to.
    """

    def __init__(self, session_source, encode):
        self._session_source = session_source
        self._encode = encode

    def add_text_prototype(self, phrase):
        phrase = (phrase or "").strip()
        if not phrase:
            return False
        session = self._session_source()
        if phrase not in session.phrases():
            session.add_phrase(phrase, self._encode(phrase))
        return True

    def remove_text_prototype(self, phrase):
        self._session_source().remove_phrase(phrase)

    def text_prototype_phrases(self):
        return self._session_source().phrases()


class PromptSessionPanel(QGroupBox):
    """Rows, summary, warnings and the fixed buttons for one `PromptSession`.

    The session is read through `session_source` on every refresh, so a dialog
    may replace its session object without telling the panel.

    Signals:
        edited: The user toggled, removed or cleared examples here. Not emitted
            for changes made from outside and then shown with `refresh`.
        saveRequested, loadRequested, inspectRequested: The fixed buttons, for the
            dialog to carry out.
    """

    edited = pyqtSignal()
    saveRequested = pyqtSignal()
    loadRequested = pyqtSignal()
    inspectRequested = pyqtSignal()

    def __init__(self, session_source, stem_source=None, imgsz_source=None,
                 title="Prompt", empty_text="Empty.", parent=None):
        """
        Args:
            session_source (callable): Returns the `PromptSession` to show.
            stem_source (callable, optional): Returns the checkpoint stem a run
                would use, for the other-model warning.
            imgsz_source (callable, optional): Returns the image size a run would
                use, for the image-size warnings.
            title (str): Group box title.
            empty_text (str): What to say while there are no examples.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(title, parent)
        self._session_source = session_source
        self._stem_source = stem_source
        self._imgsz_source = imgsz_source
        self.empty_text = empty_text

        layout = QVBoxLayout(self)

        self.summary_label = fixed_width_text(QLabel())
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.list_widget = QListWidget()
        self.list_widget.setToolTip(
            "Examples, under where they came from.\n"
            "+ finds more like it, - finds fewer like it.\n"
            "Untick an example to leave it out of the prompt.")
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget, 1)

        self.warning_label = fixed_width_text(QLabel())
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b36b00;")
        layout.addWidget(self.warning_label)

        # Dialog-specific actions go here, above the buttons every dialog shares.
        self.host_layout = QHBoxLayout()
        layout.addLayout(self.host_layout)

        buttons = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected example (a grouped row removes all of it).")
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Remove every example.")
        self.clear_button.clicked.connect(self.clear)
        buttons.addWidget(self.clear_button)

        self.save_button = QPushButton("Save...")
        self.save_button.setToolTip("Save the prompt, threshold included, as a .npz file.")
        self.save_button.clicked.connect(self.saveRequested.emit)
        buttons.addWidget(self.save_button)

        self.load_button = QPushButton("Load...")
        self.load_button.setToolTip("Load a saved prompt, or a prompt-embedding .npz file.")
        self.load_button.clicked.connect(self.loadRequested.emit)
        buttons.addWidget(self.load_button)

        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.setToolTip(
            "Plot the examples, and rank phrases against them (Text Alignment).\n"
            "A phrase kept there becomes a text example.")
        self.inspect_button.clicked.connect(self.inspectRequested.emit)
        buttons.addWidget(self.inspect_button)
        layout.addLayout(buttons)

        self.refresh()

    @property
    def session(self):
        return self._session_source()

    def add_host_widget(self, widget):
        """Put a dialog's own button in the slot above the fixed buttons."""
        self.host_layout.addWidget(widget)

    # --- showing ------------------------------------------------------------------------------------------------------

    def refresh(self):
        """Redraw headings, rows, summary and warnings from the session."""
        session = self.session
        stem = self._stem_source() if self._stem_source else None
        imgsz = self._imgsz_source() if self._imgsz_source else None

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for section, title, hint in SECTIONS:
            positives = [p for p in session.positives if section_of(p) == section]
            negatives = [p for p in session.negatives if section_of(p) == section]
            if not positives and not negatives:
                continue
            self._add_heading(f"{title}: {section_counts(section, len(positives), len(negatives))}", hint)
            for negative, prototypes in ((False, positives), (True, negatives)):
                for row in group_rows(prototypes):
                    self._add_row(row, negative, imgsz)
        self.list_widget.blockSignals(False)

        if session.is_empty():
            self.summary_label.setText(self.empty_text)
        else:
            self.summary_label.setText(session.summary())

        warnings = session_warnings(session, stem=stem, imgsz=imgsz)
        self.warning_label.setText("\n".join(warnings))
        self.warning_label.setVisible(bool(warnings))

        has_rows = not session.is_empty()
        self.remove_button.setEnabled(has_rows)
        self.clear_button.setEnabled(has_rows)
        self.save_button.setEnabled(bool(session.positives))

    def _add_heading(self, text, hint):
        item = QListWidgetItem(text)
        # Not selectable or checkable: a heading is not an example.
        item.setFlags(Qt.ItemIsEnabled)
        font = QFont(item.font())
        font.setBold(True)
        item.setFont(font)
        item.setToolTip(hint)
        self.list_widget.addItem(item)

    def _add_row(self, row, negative, imgsz):
        item = QListWidgetItem(row_text(row, negative, imgsz=imgsz))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        enabled = [p.enabled for p in row]
        if all(enabled):
            state = Qt.Checked
        elif any(enabled):
            state = Qt.PartiallyChecked
        else:
            state = Qt.Unchecked
        item.setCheckState(state)
        item.setData(Qt.UserRole, [p.uid for p in row])
        item.setToolTip(row_tooltip(row, negative))
        self.list_widget.addItem(item)

    def _items(self, headings):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if (item.data(Qt.UserRole) is None) == headings:
                yield i, item

    def rows(self):
        """The example rows' texts, top to bottom, without headings."""
        return [item.text() for _, item in self._items(headings=False)]

    def headings(self):
        """The section headings, top to bottom."""
        return [item.text() for _, item in self._items(headings=True)]

    def select_row(self, index):
        """Make the `index`-th example row (headings not counted) the current one."""
        row = [i for i, _ in self._items(headings=False)][index]
        self.list_widget.setCurrentRow(row)

    def row_item(self, index):
        """The `index`-th example row's item (headings not counted)."""
        return [item for _, item in self._items(headings=False)][index]

    def warnings(self):
        text = self.warning_label.text()
        return text.split("\n") if text else []

    # --- editing ------------------------------------------------------------------------------------------------------

    def _on_item_changed(self, item):
        on = item.checkState() != Qt.Unchecked
        for uid in item.data(Qt.UserRole) or []:
            self.session.set_enabled(uid, on)
        self.refresh()
        self.edited.emit()

    def remove_selected(self):
        """Remove the selected row's examples."""
        item = self.list_widget.currentItem()
        uids = item.data(Qt.UserRole) if item is not None else None
        if not uids:  # nothing selected, or a heading
            return
        for uid in uids:
            self.session.remove(uid)
        self.refresh()
        self.edited.emit()

    def clear(self):
        """Remove every example."""
        self.session.clear()
        self.refresh()
        self.edited.emit()


class VPEVisualizationDialog(QDialog):
    """
    Dialog for visualizing VPE embeddings, now including K-prototypes.
    """
    def __init__(self, vpe_list_with_source, final_vpe=None, prototypes=None,
                 clustering_performed=False, k_value=0, model=None, label_names=None,
                 prompt_store=None, parent=None):
        """
        Initialize the dialog.

        Args:
            vpe_list_with_source (list): List of (VPE tensor, source_str) tuples for raw VPEs.
            final_vpe (torch.Tensor, optional): The final (averaged) VPE.
            prototypes (list, optional): List of K-prototype VPE tensors (cluster centroids).
            clustering_performed (bool): Whether clustering was performed.
            k_value (int): The K value used for clustering.
            model (YOLOE, optional): The model the VPEs came from. Text embeddings are
                checkpoint-specific, so the alignment tab needs this exact model; without
                it the tab still appears, explaining why it cannot run.
            label_names (list[str], optional): Project labels to offer as candidate phrases.
            prompt_store (optional): Where a phrase the user likes is kept as a prompt,
                rather than only copied.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("VPE Visualization")
        self.resize(1000, 1000)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Store the VPEs and clustering info
        self.vpe_list_with_source = vpe_list_with_source
        self.final_vpe = final_vpe
        self.prototypes = prototypes if prototypes else []
        self.clustering_performed = clustering_performed
        self.k_value = k_value

        # Create the layout
        layout = QVBoxLayout(self)

        # The scatter answers "are my references consistent?"; the alignment tab
        # answers "what would I type to find them again?". Same embeddings, two
        # questions, so they share a window rather than a layout.
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Embedding space tab ---
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        # Create the plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("PCA Visualization of Visual Prompt Embeddings", color="#000000", size="10pt")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_layout.addWidget(self.plot_widget)
        plot_layout.addSpacing(20)

        # Add information label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.info_label)

        self.tabs.addTab(plot_tab, "Embedding Space")

        # --- Text alignment tab ---
        self.alignment_widget = PromptAlignmentWidget(model,
                                                      vpe_list_with_source,
                                                      label_names=label_names,
                                                      prompt_store=prompt_store,
                                                      parent=self)
        self.tabs.addTab(self.alignment_widget, "Text Alignment")

        # An empty scatter is a blank first impression, and the only reason to be
        # here with no embeddings is to work from text.
        if not vpe_list_with_source:
            self.tabs.setCurrentWidget(self.alignment_widget)

        # Create the button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Visualize the VPEs
        self.visualize_vpes()

    def visualize_vpes(self):
        """
        Apply PCA to all VPEs (raw, prototypes, final) and visualize them.
        """
        if not self.vpe_list_with_source:
            self.info_label.setText("No VPEs available to visualize.")
            return

        # 1. Collect all numpy arrays for PCA transformation
        raw_vpe_arrays = [vpe.detach().cpu().numpy().squeeze() for vpe, source in self.vpe_list_with_source]
        prototype_arrays = [p.detach().cpu().numpy().squeeze() for p in self.prototypes]

        all_arrays_for_pca = raw_vpe_arrays + prototype_arrays

        final_vpe_array = None
        if self.final_vpe is not None:
            final_vpe_array = self.final_vpe.detach().cpu().numpy().squeeze()
            all_arrays_for_pca.append(final_vpe_array)

        if len(all_arrays_for_pca) < 2:
            self.info_label.setText("At least 2 VPEs are needed for PCA visualization.")
            return

        # 2. Apply PCA
        all_vpes_stacked = np.vstack(all_arrays_for_pca)
        pca = PCA(n_components=2)
        vpes_2d = pca.fit_transform(all_vpes_stacked)

        # 3. Plot the results
        self.plot_widget.clear()
        self.plot_widget.addLegend(colCount=3)

        # Slicing indices
        num_raw = len(raw_vpe_arrays)
        num_prototypes = len(prototype_arrays)

        # Determine if each raw VPE is effectively a prototype (k==0 or k>=N)
        each_vpe_is_prototype = (self.k_value == 0 or self.k_value >= num_raw)

        # Plot individual raw VPEs
        colors = self.generate_distinct_colors(num_raw)
        for i, (vpe_tuple, vpe_2d) in enumerate(zip(self.vpe_list_with_source, vpes_2d[:num_raw])):
            # Use diamonds if each VPE is a prototype, circles otherwise
            symbol = 'd' if each_vpe_is_prototype else 'o'

            # If it's a prototype, add a black border
            pen = pg.mkPen(color='k', width=1.5) if each_vpe_is_prototype else None

            scatter = pg.ScatterPlotItem(
                x=[vpe_2d[0]],
                y=[vpe_2d[1]],
                brush=pg.mkColor(colors[i]),
                pen=pen,
                size=15 if not each_vpe_is_prototype else 18,
                symbol=symbol,
                name=f"{i + 1}: {vpe_tuple[1]}"
            )
            self.plot_widget.addItem(scatter)

        # Plot K-Prototypes (blue diamonds) if we have any and explicit clustering was performed
        if self.prototypes and self.clustering_performed:
            prototype_vpes_2d = vpes_2d[num_raw: num_raw + num_prototypes]
            scatter = pg.ScatterPlotItem(
                x=prototype_vpes_2d[:, 0],
                y=prototype_vpes_2d[:, 1],
                brush=pg.mkBrush(color=(0, 0, 255, 150)),
                pen=pg.mkPen(color='k', width=1.5),
                size=18,
                symbol='d',
                name=f"K-Prototypes (K={self.k_value})"
            )
            self.plot_widget.addItem(scatter)

        # Plot the final (averaged) VPE (red star)
        if final_vpe_array is not None:
            final_vpe_2d = vpes_2d[-1]
            scatter = pg.ScatterPlotItem(
                x=[final_vpe_2d[0]],
                y=[final_vpe_2d[1]],
                brush=pg.mkBrush(color='r'),
                size=20,
                symbol='star',
                name="Final VPE (Avg)"
            )
            self.plot_widget.addItem(scatter)

        # 4. Update the information label
        orig_dim = self.vpe_list_with_source[0][0].shape[-1]
        explained_variance = sum(pca.explained_variance_ratio_)

        info_text = (f"Original dimension: {orig_dim} → Reduced to 2D\n"
                     f"Total explained variance: {explained_variance:.2%}\n"
                     f"PC1: {pca.explained_variance_ratio_[0]:.2%} variance, "
                     f"PC2: {pca.explained_variance_ratio_[1]:.2%} variance\n"
                     f"Number of raw VPEs: {num_raw}\n")

        if self.clustering_performed:
            info_text += f"Clustering performed with K={self.k_value}\n"
            info_text += f"Number of prototypes: {len(self.prototypes)}"
        else:
            if self.k_value == 0:
                info_text += f"No clustering (K=0): all {num_raw} raw VPEs used as prototypes"
            else:
                info_text += f"No clustering performed (K={self.k_value} >= {num_raw}): all raw VPEs used as prototypes"

        self.info_label.setText(info_text)

    def generate_distinct_colors(self, num_colors):
        """Generates visually distinct colors.

        Deterministic: saturation and value used to come from `random.uniform`,
        so the same VPE was drawn in a different colour every time the dialog
        was opened and the plot could not be compared with itself. The golden-
        ratio hue step already separates the colours; cycling saturation and
        value over three fixed steps separates neighbouring hues further.
        """
        saturations = (1.0, 0.75, 0.6)
        values = (0.95, 0.8, 0.7)

        colors = []
        for i in range(num_colors):
            hue = (i * 0.618033988749895) % 1.0
            saturation = saturations[i % len(saturations)]
            value = values[(i // len(saturations)) % len(values)]
            r, g, b = hsv_to_rgb(hue, saturation, value)
            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            colors.append(hex_color)

        return colors
