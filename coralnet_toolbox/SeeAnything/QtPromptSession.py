"""The one See Anything prompt, and the Qt parts both See Anything dialogs share for it.

The Predictor (with its tool) and the Generator read and write the same
`PromptSession`: examples added in the tool are in the Generator's prompt, and a
calibration in the Generator is what the tool predicts from. Neither dialog lists
the examples: the tool adds them (more like this, fewer like this), Calibrate
chooses which of a label's images are in use, and Clear Prompt removes them all.
What a dialog shows is a sentence about where the examples came from, warnings,
and the same three buttons.
"""

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

from coralnet_toolbox.SeeAnything.PromptSession import (ORIGIN_ANNOTATIONS, ORIGIN_FILE,
                                                        ORIGIN_PHRASE, ORIGIN_TOOL, ORIGIN_TOOL_PHRASE,
                                                        PromptSession)


# Where examples came from, in the order they are described. The key is the
# section `section_of` puts an example in; None collects examples with no origin.
SECTION_ANNOTATIONS = ORIGIN_ANNOTATIONS
SECTION_TOOL = "tool"
SECTION_PHRASES = ORIGIN_PHRASE
SECTION_FILES = ORIGIN_FILE
SECTION_WHERE = (
    (SECTION_ANNOTATIONS, "from annotations"),
    (SECTION_TOOL, "from the tool"),
    (SECTION_PHRASES, ""),
    (SECTION_FILES, "from files"),
    (None, "from elsewhere"),
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


def level_columns(*columns):
    """Make side-by-side columns of group boxes end level with each other.

    Every box gets an equal share of whatever height its column has spare, so
    the shorter column's boxes grow a little each until both columns end at the
    same line; the taller column has nothing spare and keeps its natural heights.
    """
    for column in columns:
        for index in range(column.count()):
            column.setStretch(index, 1)


def fit_height(dialog, grow_only=False):
    """Make a dialog exactly as tall as its contents need at its current width.

    Word-wrapped labels need more height the narrower they are, so the height is
    asked of the layout for this width rather than taken from a size hint.

    Args:
        dialog (QWidget): The dialog. Its `layout` attribute may be shadowed by
            a Python attribute, so the Qt method is called through QWidget.
        grow_only (bool): Only ever make it taller, so text that grew is never
            squeezed and a dialog the user made bigger is not shrunk under them.
    """
    layout = QWidget.layout(dialog)
    if layout is None:
        return
    layout.activate()
    width = dialog.width()
    if layout.hasHeightForWidth():
        height = layout.totalHeightForWidth(width)
    else:
        height = layout.totalSizeHint().height()
    if grow_only and height <= dialog.height():
        return
    dialog.resize(width, height)


def reserve_text_height(label, sample):
    """Keep room in a word-wrapped label for `sample`, the longest text it is planned to show.

    Measured at the label's current width with the label's own layout, so text
    that changes later within that length never changes the label's height, or
    the dialog's. Call it again when the width changes. Shorter text sits at the
    top of the room kept rather than floating in the middle of it.
    """
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    width = label.width()
    if width <= 0:
        return
    current = label.text()
    label.setText(sample)
    needed = label.heightForWidth(width)
    label.setText(current)
    if needed > 0 and needed != label.minimumHeight():
        label.setMinimumHeight(needed)


def reserve_lines(label, count):
    """Keep room in a label for `count` lines of text."""
    reserve_text_height(label, "\n".join(["X"] * count))


def warning_label():
    """A word-wrapped orange label for a prompt's warnings, hidden until `show_warnings` gives it some."""
    label = fixed_width_text(QLabel())
    label.setWordWrap(True)
    label.setStyleSheet("color: #b36b00;")
    label.setVisible(False)
    return label


def show_warnings(label, warnings):
    """Put warnings in a `warning_label`, one per line, and hide it when there are none."""
    label.setText("\n".join(warnings))
    label.setVisible(bool(warnings))


def section_of(prototype):
    """The section an example is described under."""
    source = prototype.source
    if source in (ORIGIN_TOOL, ORIGIN_TOOL_PHRASE):
        return SECTION_TOOL
    if source in (ORIGIN_ANNOTATIONS, ORIGIN_PHRASE, ORIGIN_FILE):
        return source
    return None


def _plural(count, word, plural=None):
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def section_counts(section, positives, negatives):
    """"3 images", "2 examples, 1 negative", "1 phrase" for one section's examples."""
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
    """Where a prompt's examples came from, in one line, and how many of them are left out.

    E.g. "64 images from annotations (27 left out), 2 examples and 1 negative
    from the tool, 1 phrase". The counts are the examples in use: they are what
    a run uses. Examples are left out by a calibration, not by hand.
    """
    if session.is_empty():
        return "empty"

    parts = []
    for section, where in SECTION_WHERE:
        positives = [p for p in session.positives if section_of(p) == section]
        negatives = [p for p in session.negatives if section_of(p) == section]
        if not positives and not negatives:
            continue
        on_positives = sum(1 for p in positives if p.enabled)
        on_negatives = sum(1 for p in negatives if p.enabled)
        left_out = len(positives) + len(negatives) - on_positives - on_negatives
        counts = section_counts(section, on_positives, on_negatives).replace(", ", " and ") or "none"
        text = f"{counts} {where}".strip()
        if left_out:
            text += f" ({left_out} left out)"
        parts.append(text)
    return ", ".join(parts)


def session_signature(session):
    """What a prompt is made of, cheaply: its model, and each example's id, state and embedding.

    Two signatures differ when an example was added, removed, switched on or off,
    or embedded again, which is when a prediction made from the prompt is stale.
    """
    def rows(prototypes):
        return tuple((p.uid, p.enabled, id(p.embedding)) for p in prototypes)

    return session.model_stem, rows(session.positives), rows(session.negatives)


def session_warnings(session, stem=None, imgsz=None, reembeds=True):
    """Everything worth saying about a prompt before it runs.

    Args:
        session (PromptSession): The prompt.
        stem (str, optional): The checkpoint the prompt would run with.
        imgsz (int, optional): The image size the prompt would run at.
        reembeds (bool): Whether this dialog embeds examples from annotations
            again at its own image size before running. The Generator does; the
            tool uses every example as it was made.

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
            "Fewer-like-this examples only work against a visual example; with text alone they are ignored.")

    off_size = session.off_size(imgsz)
    redo = [p for p in off_size if p.can_reembed and reembeds]
    fixed = [p for p in off_size if p not in redo]
    if redo:
        warnings.append(
            f"{len(redo)} example{'s were' if len(redo) != 1 else ' was'} embedded at another "
            f"image size and will be embedded again at {imgsz} when the run starts.")
    if fixed:
        sizes = ", ".join(str(s) for s in sorted({int(p.imgsz) for p in fixed}))
        if reembeds:
            again = "cannot be embedded again"
        else:
            again = f"{'are' if len(fixed) != 1 else 'is'} not embedded again here"
        warnings.append(
            f"{len(fixed)} example{'s were' if len(fixed) != 1 else ' was'} embedded at {sizes}, "
            f"not {imgsz}, and {again}: an embedding depends on the image size, "
            f"so predict at {sizes} to use {'them' if len(fixed) != 1 else 'it'} as made.")
    return warnings


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SharedPrompt(QObject):
    """The one prompt both See Anything dialogs use.

    A dialog that edits it, or only redraws itself, calls `notify`; the other
    dialog redraws, and decides from `session_signature` whether the prompt
    actually changed.

    Signals:
        changed(object): The prompt may have changed; the argument is the dialog
            that says so, so it can ignore its own notice.
    """

    changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = PromptSession()

    def notify(self, source=None):
        self.changed.emit(source)


def shared_prompt(main_window):
    """The main window's `SharedPrompt`, made on first use.

    The main window makes it before either See Anything dialog; making it here
    too means a dialog built without one (a test, a script) still works.
    """
    prompt = getattr(main_window, 'see_anything_prompt', None)
    if prompt is None:
        prompt = SharedPrompt()
        main_window.see_anything_prompt = prompt
    return prompt


class PromptButtons(QObject):
    """Clear, Save and Load for the prompt, the same in both dialogs.

    Only the buttons: each dialog puts them in its own row of actions, so every
    button there shares the width alike. The dialog carries each action out.

    Signals:
        clearRequested, saveRequested, loadRequested: The buttons.
    """

    clearRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    loadRequested = pyqtSignal()

    def __init__(self, clear_tooltip="Remove every example.", parent=None):
        super().__init__(parent)
        self.clear_button = QPushButton("Clear Prompt")
        self.clear_button.setToolTip(clear_tooltip)
        self.clear_button.clicked.connect(self.clearRequested.emit)

        self.save_button = QPushButton("Save Prompt...")
        self.save_button.setToolTip("Save the prompt, threshold included, as a .npz file.")
        self.save_button.clicked.connect(self.saveRequested.emit)

        self.load_button = QPushButton("Load Prompt...")
        self.load_button.setToolTip("Add the examples of a saved prompt, or of a prompt-embedding .npz file.")
        self.load_button.clicked.connect(self.loadRequested.emit)

    @property
    def buttons(self):
        return [self.clear_button, self.save_button, self.load_button]

    def refresh(self, session):
        """Enable the buttons that have something to act on."""
        self.clear_button.setEnabled(not session.is_empty())
        self.save_button.setEnabled(bool(session.positives))
