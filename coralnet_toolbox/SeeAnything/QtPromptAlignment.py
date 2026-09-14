from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QSpinBox, QPushButton, QLineEdit, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QApplication, QDialogButtonBox, QListWidget, QListWidgetItem,
                             QGroupBox)

from coralnet_toolbox.QtProgressBar import ProgressBar

from coralnet_toolbox.SeeAnything.PromptAlignment import (TextEmbedder,
                                                          as_matrix,
                                                          checkpoint_stem,
                                                          margin,
                                                          promptable,
                                                          rank,
                                                          vocabulary_embeddings,
                                                          vocabulary_is_cached,
                                                          VOCABULARY_CHECKPOINT_MB)


# Candidate sources, in the order they appear in the combo box.
# Every word the model knows comes first, because that is the question the tab
# exists to answer: not "which of my labels is closest" -- the user already knows
# their labels -- but "what does this thing look like to the model".
SOURCE_VOCABULARY = "Every word the model knows (4,585)"
SOURCE_LABELS = "My project labels"
SOURCE_CUSTOM = "Words I type"


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def ensure_vocabulary(parent, embedder, stem=None, ask=True):
    """Every word the model knows, building the list once if it is not cached.

    The 4,585 names are published only inside the prompt-free checkpoints, so the
    first call downloads one to read `model.names`, encodes all of them for this
    checkpoint, and writes the result next to the weights. Measured at 79 s on a
    CPU; every call afterwards reads a 9.4 MB NPZ.

    Shared by the Generator's alignment tab and the tool's Ctrl+T suggestions so
    both offer the same vocabulary and pay for it at most once.

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


class PromptAlignmentWidget(QWidget):
    """Ranks candidate phrases against the visual prompt embeddings in hand.

    The ranking reports a handle, not an identification. A crop of a bus ranks
    `vehicle` above `bus`; what the column headings promise is "what to type",
    which is the question a text prompt actually asks.
    """

    textChosen = pyqtSignal(str)

    def __init__(self, model, vpe_entries, label_names=None, prompt_store=None, parent=None):
        """
        Args:
            model: The loaded `YOLOE` the VPEs came from. Text embeddings are
                checkpoint-specific, so this must be that same model.
            vpe_entries (list): `(tensor, source_str)` pairs, as the VPE
                visualization already collects them.
            label_names (list[str], optional): Project label names to offer as
                candidates.
            prompt_store (optional): The Generator dialog, or anything with
                `add_text_prototype`, `remove_text_prototype` and
                `text_prototype_phrases`. Supplying it is what turns a ranking
                into something the user can actually run with; without it the
                tab is read-only and phrases can only be copied.
            parent (QWidget, optional): Parent widget.
        """
        super().__init__(parent)

        self.model = model
        self.vpe_matrix = as_matrix([vpe for vpe, _ in (vpe_entries or [])])
        self.label_names = list(label_names or [])
        self.prompt_store = prompt_store
        self.embedder = TextEmbedder(model) if model is not None else None
        self.ranked = []
        self._auto_ranked = False

        self._setup_ui()
        self._refresh_kept()
        self._update_availability()

    def _setup_ui(self):
        """Build the layout."""
        layout = QVBoxLayout(self)

        self.explanation = QLabel(
            "Scores your reference annotations against every word the model knows, best first.\n"
            "The winner is whatever points at the same thing, which is not always the correct "
            "name for it and need not be one of your labels: a photo of a bus scores "
            "'vehicle' above 'bus'. Keep the winner to use it as a prompt here, or type it "
            "into the tool with Ctrl+T."
        )
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)

        controls = QGroupBox("Candidates")
        controls_layout = QVBoxLayout(controls)

        row = QHBoxLayout()
        row.addWidget(QLabel("Rank:"))

        self.source_combo = QComboBox()
        self.source_combo.addItems([SOURCE_VOCABULARY, SOURCE_LABELS, SOURCE_CUSTOM])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        row.addWidget(self.source_combo, 1)

        row.addWidget(QLabel("Show top:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 500)
        self.top_k_spin.setValue(25)
        row.addWidget(self.top_k_spin)

        self.rank_button = QPushButton("Rank")
        self.rank_button.clicked.connect(self.rank_candidates)
        row.addWidget(self.rank_button)
        controls_layout.addLayout(row)

        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText("Comma-separated words, e.g. coral, sponge, rubble, sand")
        self.custom_edit.returnPressed.connect(self.rank_candidates)
        self.custom_edit.setVisible(False)
        controls_layout.addWidget(self.custom_edit)

        layout.addWidget(controls)

        self.table = QTableWidget(0, 4)
        # "Alignment" rather than "Score": it is a similarity between this phrase
        # and your reference crops, and reads as a detection confidence otherwise.
        # It is not one, and does not predict one -- measured on a bus crop,
        # 'vehicle' aligns at 0.438 and finds one object, while 'person' aligns at
        # 0.253 and finds eight.
        self.table.setHorizontalHeaderLabels(
            ["Phrase", "Alignment", "Worst reference", "Spread"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemDoubleClicked.connect(self._on_row_activated)
        layout.addWidget(self.table, 1)

        self.kept_group = QGroupBox("Prompts kept from text")
        kept_layout = QVBoxLayout(self.kept_group)

        kept_row = QHBoxLayout()
        self.keep_button = QPushButton("Keep the selected phrase")
        self.keep_button.setToolTip(
            "Adds the phrase to this run's prompts, alongside your reference images. "
            "Both are the same kind of embedding, so you can use text, images, or both."
        )
        self.keep_button.clicked.connect(self.keep_selected)
        kept_row.addWidget(self.keep_button)

        self.drop_button = QPushButton("Remove")
        self.drop_button.clicked.connect(self.drop_selected_kept)
        kept_row.addWidget(self.drop_button)

        self.clear_button = QPushButton("Remove all")
        self.clear_button.clicked.connect(self.clear_kept)
        kept_row.addWidget(self.clear_button)
        kept_layout.addLayout(kept_row)

        self.kept_list = QListWidget()
        self.kept_list.setMaximumHeight(90)
        kept_layout.addWidget(self.kept_list)

        layout.addWidget(self.kept_group)
        self.kept_group.setVisible(self.prompt_store is not None)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def _update_availability(self):
        """Enable only what can be done right now, and say what is missing.

        Ranking and keeping have different requirements. Ranking needs something
        to rank against; keeping a phrase you typed yourself does not, so a user
        with no references at all can still set up a text-only run.
        """
        can_encode = self.model is not None and promptable(self.model)
        has_vpes = self.vpe_matrix.numel() > 0

        self.rank_button.setEnabled(can_encode and has_vpes)
        self.source_combo.setEnabled(can_encode and has_vpes)
        self.top_k_spin.setEnabled(can_encode and has_vpes)
        self.keep_button.setEnabled(can_encode and self.prompt_store is not None)

        if self.model is None:
            self.info_label.setText("Load a model to rank text against your references.")
        elif not can_encode:
            self.info_label.setText(
                "This model's head has been fused, so it can no longer encode text.")
        elif not has_vpes:
            # The source combo is disabled here, so put the one usable source in
            # front of the user rather than leaving them with a dead control.
            self.source_combo.setCurrentText(SOURCE_CUSTOM)
            self.custom_edit.setVisible(True)
            self.info_label.setText(
                "No VPEs to rank against. You can still type a phrase below and keep it "
                "as a prompt. Ranking needs reference embeddings; using text does not.")
        else:
            count = self.vpe_matrix.shape[0]
            self.info_label.setText(
                f"Ready: {count} visual prompt embedding{'s' if count != 1 else ''} to match against."
            )

    def showEvent(self, event):
        """Answer the question on arrival when it is free to answer.

        "What does the model think this is" should be on screen when the tab
        opens, not behind a button press. Only when the word list is already
        cached, though: opening a tab must never silently start a download.
        """
        super().showEvent(event)
        if self._auto_ranked:
            return
        self._auto_ranked = True
        self.rank_if_free()

    def rank_if_free(self):
        """Rank the full vocabulary if doing so costs nothing, else say what it costs."""
        if (self.embedder is None
                or self.vpe_matrix.numel() == 0
                or not promptable(self.model)
                or self.source_combo.currentText() != SOURCE_VOCABULARY):
            return

        if vocabulary_is_cached(checkpoint_stem(self.model)):
            self.rank_candidates()
        else:
            self.info_label.setText(
                "Press Rank to score your references against every word the model knows. "
                "The word list is built once for this checkpoint (a download and about a "
                "minute), then cached."
            )

    def _on_source_changed(self, source):
        """Show the free-text box only when the user is supplying the words."""
        self.custom_edit.setVisible(source == SOURCE_CUSTOM)

    def _on_row_activated(self, item):
        """Double-click puts a phrase on the clipboard and announces it."""
        text = self.table.item(item.row(), 0).text()
        QApplication.clipboard().setText(text)
        self.info_label.setText(f"Copied '{text}' to the clipboard. Use it as a text prompt with Ctrl+T.")
        self.textChosen.emit(text)

    # ------------------------------------------------------------------
    # Keeping phrases as prompts
    # ------------------------------------------------------------------

    def selected_phrase(self):
        """The phrase the Keep button would act on.

        The selected row, or the top-ranked one when nothing is selected, so the
        common case -- rank, then keep the winner -- takes one click rather than
        two. With nothing ranked at all it falls back to whatever is typed in the
        free-text box, which is what makes a text-only run possible.

        Returns:
            str | None: The phrase, or None if there is nothing to keep.
        """
        row = self.table.currentRow()
        if row < 0:
            row = 0
        item = self.table.item(row, 0)
        if item is not None:
            return item.text()

        typed = self.custom_candidates()
        return typed[0] if typed else None

    def keep_selected(self):
        """Add the selected phrase to the prompts this run will use."""
        if self.prompt_store is None:
            return

        phrase = self.selected_phrase()
        if not phrase:
            self.info_label.setText("Rank some phrases first, then keep the one you want.")
            return

        try:
            self.prompt_store.add_text_prototype(phrase)
        except Exception as e:
            QMessageBox.critical(self, "Could Not Keep That Phrase", f"An error occurred: {e}")
            return

        self._refresh_kept()
        self.info_label.setText(
            f"'{phrase}' is now one of this run's prompts. "
            f"It works alongside your reference images: text, images, or both."
        )

    def drop_selected_kept(self):
        """Remove the highlighted kept phrase."""
        if self.prompt_store is None:
            return
        item = self.kept_list.currentItem()
        if item is None:
            return
        self.prompt_store.remove_text_prototype(item.text())
        self._refresh_kept()

    def clear_kept(self):
        """Remove every kept phrase."""
        if self.prompt_store is None:
            return
        for phrase in list(self.prompt_store.text_prototype_phrases()):
            self.prompt_store.remove_text_prototype(phrase)
        self._refresh_kept()

    def kept_phrases(self):
        """The phrases currently held as prompts."""
        if self.prompt_store is None:
            return []
        return list(self.prompt_store.text_prototype_phrases())

    def _refresh_kept(self):
        """Redraw the kept list and enable only what can be acted on."""
        if self.prompt_store is None:
            return

        phrases = self.kept_phrases()
        self.kept_list.clear()
        self.kept_list.addItems(phrases)

        self.drop_button.setEnabled(bool(phrases))
        self.clear_button.setEnabled(bool(phrases))
        self.kept_group.setTitle(
            "Prompts kept from text" if not phrases
            else f"Prompts kept from text ({len(phrases)})"
        )

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def custom_candidates(self):
        """The words typed into the free-text box, split and cleaned."""
        raw = self.custom_edit.text().replace("\n", ",")
        return [part.strip() for part in raw.split(",") if part.strip()]

    def candidates(self):
        """The phrases to rank, for the current source.

        Returns:
            list[str] | None: Phrases, or None if the user cancelled or the
            source could not be prepared.
        """
        source = self.source_combo.currentText()

        if source == SOURCE_LABELS:
            if not self.label_names:
                QMessageBox.information(self, "No Labels",
                                        "This project has no labels to rank.")
                return None
            return self.label_names

        if source == SOURCE_CUSTOM:
            words = self.custom_candidates()
            if not words:
                QMessageBox.information(self, "Nothing to Rank",
                                        "Type one or more comma-separated words first.")
                return None
            return words

        return self._vocabulary_candidates()

    def _vocabulary_candidates(self):
        """Every word the model knows, built once if it is not cached yet.

        Returns:
            list[str] | None: The vocabulary, or None if the user declined the
            one-time build or it failed.
        """
        return ensure_vocabulary(self, self.embedder, stem=checkpoint_stem(self.model))

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_candidates(self):
        """Score the current candidate list and fill the table."""
        if self.embedder is None or self.vpe_matrix.numel() == 0:
            return

        texts = self.candidates()
        if not texts:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            text_matrix = self.embedder.encode(texts)
            self.ranked = rank(self.vpe_matrix, text_matrix, texts,
                               top_k=self.top_k_spin.value())
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Could Not Rank", f"An error occurred: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._fill_table(self.ranked, len(texts))

    def _fill_table(self, ranked, considered):
        """Render the ranking and summarize how decisive it was."""
        self.table.setRowCount(len(ranked))
        for row, entry in enumerate(ranked):
            self.table.setItem(row, 0, QTableWidgetItem(entry['text']))
            for column, key in enumerate(('score', 'lowest', 'spread'), start=1):
                item = QTableWidgetItem(f"{entry[key]:.3f}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)

        if not ranked:
            self.info_label.setText("Nothing scored.")
            return

        gap = margin(ranked)
        best = ranked[0]
        verdict = "a clear winner" if gap >= 0.05 else "no clear winner; any of the top few would serve"
        self.info_label.setText(
            f"Ranked {considered} phrase{'s' if considered != 1 else ''}: "
            f"'{best['text']}' at {best['score']:.3f}, {gap:.3f} ahead of the runner-up ({verdict}). "
            f"Read the order and the gap, not the number: these are not detection "
            f"confidences and do not predict one. A large spread means your references "
            f"disagree about that phrase. Double-click a row to copy it."
        )


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
            "the closest. The best phrase is the one that finds the object, not necessarily "
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
