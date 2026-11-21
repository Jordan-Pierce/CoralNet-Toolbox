from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox,
                             QApplication, QDialogButtonBox, QListWidget, QListWidgetItem)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.SeeAnything.PromptAlignment import (checkpoint_stem,
                                                          margin,
                                                          vocabulary_embeddings,
                                                          vocabulary_is_cached,
                                                          VOCABULARY_CHECKPOINT_MB)


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def ensure_vocabulary(parent, embedder, stem=None, ask=True):
    """Every word the model knows, building the list once if it is not cached.

    The 4,585 names are published only inside the prompt-free checkpoints, so the
    first call downloads one to read `model.names`, encodes all of them for this
    checkpoint, and writes the result next to the weights. Measured at 79 s on a
    CPU; every call afterwards reads a 9.4 MB NPZ.

    Used by the tool's Ctrl+T suggestions, which pay for the list at most once.

    Args:
        parent (QWidget): Parent for the confirmation and progress dialogs.
        embedder (TextEmbedder): Bound to the model that will do the ranking.
        stem (str, optional): Checkpoint stem; defaults to the embedder's model.
        ask (bool): Confirm before the one-time build. False skips straight to it.

    Returns:
        list[str] | None: The vocabulary, or None if the user declined or it failed.
    """
    stem = stem or checkpoint_stem(embedder.model)

    if not vocabulary_is_cached(stem) and ask:
        answer = QMessageBox.question(
            parent,
            "Build the Word List?",
            f"To score a crop against every word the model knows, that word list has to "
            f"be built once for this checkpoint. It is published only inside a prompt-free "
            f"checkpoint, so this downloads one (about {VOCABULARY_CHECKPOINT_MB} MB) and "
            f"encodes all 4,585 words.\n\n"
            f"That takes about a minute. It is cached afterwards, and every ranking from "
            f"then on is instant.\n\nBuild it now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return None

    progress_bar = ProgressBar(parent, title="Building the Word List")
    progress_bar.show()
    progress_bar.set_busy_mode("Fetching the word list...")
    QApplication.setOverrideCursor(Qt.WaitCursor)

    started = {'total': 0}

    def report(done, total):
        if started['total'] != total:
            started['total'] = total
            progress_bar.start_progress(total)
        progress_bar.set_value(done)

    try:
        names, _ = vocabulary_embeddings(embedder, stem=stem, progress=report)
        return names
    except Exception as e:
        QMessageBox.critical(parent, "Word List Unavailable", str(e))
        return None
    finally:
        QApplication.restoreOverrideCursor()
        progress_bar.stop_progress()
        progress_bar.close()


def project_label_names(label_window):
    """Every project label name, as candidate phrases for text alignment.

    Both spellings go in when they differ. Which of "Porit" and "Porites
    massive" works better as a YOLOE prompt is exactly what the ranking is being
    asked, so neither is chosen here. "Review" is left out: it is a placeholder,
    not a thing anyone is looking for.

    Args:
        label_window: The main window's label window, or None.

    Returns:
        list[str]: Sorted, de-duplicated names.
    """
    names = []
    for label in getattr(label_window, 'labels', None) or []:
        short = getattr(label, 'short_label_code', None)
        if not short or short == 'Review':
            continue
        names.append(short)
        long = getattr(label, 'long_label_code', None)
        if long and long != short:
            names.append(long)

    return sorted(dict.fromkeys(names))


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TextPromptDialog(QDialog):
    """Ctrl+T's input box, with the option to ask the model what to type.

    A user who has already drawn a box knows what they want but not what YOLOE
    calls it. `suggest` closes that gap: it returns ranked phrases for the boxes
    on screen, any of which can be clicked straight into the entry field.
    """

    def __init__(self, current_text="", suggest=None, parent=None):
        """
        Args:
            current_text (str): Phrase to start from.
            suggest (callable, optional): Returns `list[dict]` as `rank` does,
                for whatever is currently drawn. None disables suggestions.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Text Prompt")
        self.resize(460, 380)

        self._suggest = suggest

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Describe what to find (leave empty to go back to box prompts):"))

        self.edit = QLineEdit(current_text or "")
        self.edit.returnPressed.connect(self.accept)
        layout.addWidget(self.edit)

        self.suggest_button = QPushButton("Suggest from the boxes I drew")
        self.suggest_button.setToolTip(
            "Scores the boxes currently drawn against every word the model knows and lists "
            "the closest.\n"
            "The best phrase is the one that finds the object, not necessarily "
            "its real name, and not necessarily one of your labels."
        )
        self.suggest_button.clicked.connect(self.populate_suggestions)
        self.suggest_button.setEnabled(suggest is not None)
        layout.addWidget(self.suggest_button)

        self.suggestions = QListWidget()
        self.suggestions.itemClicked.connect(self._on_suggestion_clicked)
        self.suggestions.itemDoubleClicked.connect(self._on_suggestion_activated)
        layout.addWidget(self.suggestions, 1)

        self.info_label = QLabel(
            "" if suggest is not None else "Draw a box first to get suggestions."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def value(self):
        """The phrase the user settled on, stripped."""
        return self.edit.text().strip()

    def populate_suggestions(self):
        """Ask the caller to rank phrases for what is currently drawn."""
        if self._suggest is None:
            return

        self.suggestions.clear()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ranked = self._suggest() or []
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.info_label.setText(f"Could not suggest anything: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not ranked:
            self.info_label.setText("Nothing to suggest from. Draw a box first.")
            return

        for entry in ranked:
            item = QListWidgetItem(f"{entry['text']}   ({entry['score']:.3f})")
            item.setData(Qt.UserRole, entry['text'])
            self.suggestions.addItem(item)

        gap = margin(ranked)
        self.info_label.setText(
            f"Closest match '{ranked[0]['text']}', {gap:.3f} ahead of the next. "
            f"Click one to use it. These are handles, not names: the word that finds the "
            f"object is not always what it is called."
        )

    def _on_suggestion_clicked(self, item):
        """Single click loads the phrase into the entry field."""
        self.edit.setText(item.data(Qt.UserRole))

    def _on_suggestion_activated(self, item):
        """Double click accepts it outright."""
        self.edit.setText(item.data(Qt.UserRole))
        self.accept()
