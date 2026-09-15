import itertools
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from coralnet_toolbox.SeeAnything.PromptAlignment import as_matrix


# Positive example kinds. "imported" is an embedding read from a file that did not
# say where it came from; every such file this toolbox has ever written held VPEs.
KIND_BOXES = "boxes"
KIND_TEXT = "text"
KIND_DETECTION = "detection"
KIND_IMPORTED = "imported"

VISUAL_KINDS = frozenset({KIND_BOXES, KIND_DETECTION, KIND_IMPORTED})
POSITIVE_KINDS = frozenset({KIND_BOXES, KIND_TEXT, KIND_DETECTION, KIND_IMPORTED})

# Where an example came from, as `Prototype.origin["source"]`. It decides whether
# the example can be embedded again -- only annotations still exist to re-read --
# and which rows the session panel shows as one.
ORIGIN_ANNOTATIONS = "annotations"  # the Generator's "Add from annotations"
ORIGIN_TOOL = "tool"                # boxes or a detection on a tool work area
ORIGIN_TOOL_PHRASE = "tool-phrase"  # the tool's one Ctrl+T phrase
ORIGIN_PHRASE = "phrase"            # a phrase added in the Generator or Inspect
ORIGIN_FILE = "file"                # a plain prompt-embedding file

# Written into session files so a reader can tell one from a plain prompt file.
SESSION_FORMAT = "coralnet-prompt-session/1"

_uids = itertools.count(1)


def default_origin(kind):
    """The origin an example of this kind had before origins were recorded.

    Session files written before then came only from the tool, and held at most
    one phrase -- the tool's.
    """
    if kind == KIND_TEXT:
        return {"source": ORIGIN_TOOL_PHRASE}
    if kind == KIND_IMPORTED:
        return {"source": ORIGIN_FILE}
    return {"source": ORIGIN_TOOL}


def _as_vector(embedding):
    """One embedding as a (D,) L2-normalized CPU float32 vector.

    Accepts the (1, 1, D) tensors `get_vpe` returns, the (1, D) rows
    `TextEmbedder.encode` returns, and bare (D,) vectors.

    Raises:
        ValueError: If the tensor holds more or fewer than one vector.
    """
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(np.asarray(embedding))
    matrix = as_matrix([embedding])
    if matrix.shape[0] != 1:
        raise ValueError(f"Expected one embedding, got {matrix.shape[0]}.")
    return matrix[0]


@dataclass(eq=False)
class Prototype:
    """One example in a session: a single embedding and what it came from.

    `imgsz` is the image size a visual example was embedded at. It is per example
    rather than per session because examples now arrive from different places, and
    it changes the vector: one set of boxes embedded at 640 and at 1024 had cosine
    0.926. Phrases do not depend on it and leave it None.

    `origin` says where the example came from, e.g.
    `{"source": "annotations", "image": path, "label": "Porites"}`.
    """

    kind: str
    embedding: torch.Tensor
    label: str
    enabled: bool = True
    uid: int = field(default_factory=lambda: next(_uids))
    imgsz: int = None
    origin: dict = None

    @property
    def is_visual(self):
        """True for examples taken from pixels; False for phrases."""
        return self.kind in VISUAL_KINDS

    @property
    def source(self):
        """`origin["source"]`, or None when the origin is unknown."""
        return (self.origin or {}).get("source")

    @property
    def can_reembed(self):
        """Whether the pixels this came from can still be read: only annotations can.

        A tool example was embedded on a work-area crop that no longer exists.
        """
        return self.is_visual and self.source == ORIGIN_ANNOTATIONS

    @property
    def group_key(self):
        """Examples sharing a key are one row in the panel: one image's annotations of one label."""
        if self.source != ORIGIN_ANNOTATIONS:
            return None
        return (self.origin.get("image"), self.origin.get("label"))

    def copy(self):
        """An independent copy -- same uid, so a UI row still finds it."""
        return Prototype(self.kind, self.embedding.clone(), self.label, self.enabled, self.uid,
                         imgsz=self.imgsz, origin=dict(self.origin) if self.origin else None)


def new_prototype(kind, embedding, label, imgsz=None, origin=None):
    """An example not yet in any session, its embedding normalized as a session stores it.

    For building a group of examples to hand to `PromptSession.replace_group`.
    """
    return Prototype(kind, _as_vector(embedding), label, imgsz=imgsz, origin=origin or default_origin(kind))


def _is_tool_phrase(prototype):
    """A phrase set with the tool's Ctrl+T. One with no origin predates origins, and came from there."""
    return prototype.kind == KIND_TEXT and prototype.source in (None, ORIGIN_TOOL_PHRASE)


def _same_example(a, b):
    if a.uid == b.uid:
        return True
    return (a.kind == b.kind and a.label == b.label
            and a.embedding.shape == b.embedding.shape
            and bool(torch.equal(a.embedding, b.embedding)))


def group_rows(prototypes):
    """Split examples into panel rows: one per example, one per annotation group.

    Groups keep the position of their first member.

    Returns:
        list[list[Prototype]]: The rows, in order.
    """
    rows, by_key = [], {}
    for prototype in prototypes:
        key = prototype.group_key
        if key is None:
            rows.append([prototype])
        elif key in by_key:
            by_key[key].append(prototype)
        else:
            by_key[key] = [prototype]
            rows.append(by_key[key])
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# Functions shared by the tool and the Generator
# ----------------------------------------------------------------------------------------------------------------------


def decoys_can_compete(positives):
    """Whether decoy classes will do anything against these positives.

    Args:
        positives: Iterable of `Prototype`, or of `(is_visual)` booleans.

    Returns:
        bool: True when at least one positive came from pixels.
    """
    for positive in positives:
        visual = positive.is_visual if isinstance(positive, Prototype) else bool(positive)
        if visual:
            return True
    return False


def stack_classes(positives, negatives=()):
    """Turn examples into the class embeddings YOLOE predicts with.

    The one function both the tool and the Generator call, so a session produces
    the same tensor wherever it runs.

    Args:
        positives: Iterable of embedding tensors, any of (D,), (1, D), (1, N, D).
        negatives: Iterable of embedding tensors for decoy classes.

    Returns:
        tuple[list[str], torch.Tensor, int]: Class names, a (1, P + N, D) CPU
        float32 tensor with positives first, and P -- the number of positive
        classes. Every class index at or above P is a decoy.

    Raises:
        ValueError: If there are no positives.
    """
    positive_matrix = as_matrix(list(positives))
    if positive_matrix.numel() == 0:
        raise ValueError("A prompt needs at least one positive example.")

    negative_matrix = as_matrix(list(negatives))
    n_positive = positive_matrix.shape[0]

    names = [f"object{i}" for i in range(n_positive)]
    if negative_matrix.numel():
        names += [f"decoy{j}" for j in range(negative_matrix.shape[0])]
        stacked = torch.cat([positive_matrix, negative_matrix], dim=0)
    else:
        stacked = positive_matrix

    return names, stacked.unsqueeze(0), n_positive


def keep_positive_detections(result, n_positive):
    """Drop every detection a decoy class won.

    Must run before anything collapses class IDs: once they are all 0, a decoy's
    detection is indistinguishable from a real one.

    Args:
        result: An ultralytics `Results`, or None.
        n_positive (int): Class indices below this are positives.

    Returns:
        The same `Results` when nothing needed dropping, otherwise a filtered one
        (boxes and masks follow the same index).
    """
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return result
    keep = boxes.cls < n_positive
    if bool(keep.all()):
        return result
    return result[keep]


def collapse_to_one_class(result, name):
    """Set every detection's class to 0, named `name`.

    See Anything is single-class by design; several positive classes are an
    implementation detail of the prompt, never something the user labels.

    Args:
        result: An ultralytics `Results`, or None.
        name (str): The one class name to report.

    Returns:
        The same `Results`, modified in place.
    """
    boxes = getattr(result, 'boxes', None)
    if boxes is not None and len(boxes) > 0:
        data = boxes.data.clone()
        data[:, 5] = 0
        result.boxes = type(boxes)(data, boxes.orig_shape)
    if result is not None:
        result.names = {0: name}
    return result


def stem_from_path(model_path):
    """The prompt-embedding stem ultralytics would bind a checkpoint to.

    Mirrors `YOLOE._prompt_embedding_model` -- the file stem with `-seg` removed --
    so a stem can be checked before the model is loaded.
    """
    if not model_path:
        return None
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    return stem[:-4] if stem.endswith("-seg") else stem


# ----------------------------------------------------------------------------------------------------------------------
# The session
# ----------------------------------------------------------------------------------------------------------------------


class PromptSession:
    """Positive and negative examples plus a threshold, for one checkpoint.

    Embeddings are only meaningful to the checkpoint that made them -- text goes
    through that model's own `reprta` head, crops through its own `SAVPE` -- so a
    session records its model stem and refuses to be used with another.
    """

    def __init__(self, model_stem=None):
        self.model_stem = model_stem
        self.positives = []
        self.negatives = []
        self.confidence = None
        # Image size the embeddings were extracted at. It changes them: one set of
        # boxes embedded at 640 and at 1024 gave vectors with cosine 0.926, and
        # detection confidences moved by up to 0.07. A run should predict at it.
        self.imgsz = None

    # --- building -------------------------------------------------------------------------------------------------

    def add_positive(self, kind, embedding, label, imgsz=None, origin=None):
        """Add an example of what to find.

        Returns:
            Prototype: The example added.
        """
        if kind not in POSITIVE_KINDS:
            raise ValueError(f"Unknown example kind '{kind}'.")
        prototype = Prototype(kind, _as_vector(embedding), label, imgsz=imgsz,
                              origin=origin or default_origin(kind))
        self.positives.append(prototype)
        return prototype

    def add_negative(self, embedding, label, imgsz=None, origin=None):
        """Add an example of what not to find.

        Returns:
            Prototype: The example added.
        """
        prototype = Prototype(KIND_DETECTION, _as_vector(embedding), label, imgsz=imgsz,
                              origin=origin or default_origin(KIND_DETECTION))
        self.negatives.append(prototype)
        return prototype

    def set_text(self, phrase, embedding=None):
        """Replace the tool's phrase, or remove it with None.

        The interactive tool holds one phrase at a time (Ctrl+T), so setting a
        new one replaces the last rather than piling up. Only the phrase the tool
        set is replaced: phrases added in the Generator are separate rows, so
        loading a Generator session into the tool does not wipe them.

        Returns:
            Prototype | None: The new text example, if one was set.
        """
        self.positives = [p for p in self.positives if not _is_tool_phrase(p)]
        phrase = (phrase or "").strip()
        if not phrase or embedding is None:
            return None
        return self.add_positive(KIND_TEXT, embedding, phrase,
                                 origin={"source": ORIGIN_TOOL_PHRASE})

    def add_phrase(self, phrase, embedding):
        """Add a phrase as one more positive, alongside any others.

        Returns:
            Prototype | None: The phrase's example (the existing one if it is
            already here), or None for an empty phrase.
        """
        phrase = (phrase or "").strip()
        if not phrase:
            return None
        for prototype in self.positives:
            if prototype.kind == KIND_TEXT and prototype.source == ORIGIN_PHRASE and prototype.label == phrase:
                return prototype
        return self.add_positive(KIND_TEXT, embedding, phrase, origin={"source": ORIGIN_PHRASE})

    def remove_phrase(self, phrase):
        """Remove a phrase added with `add_phrase`.

        Returns:
            bool: True if something was removed.
        """
        before = len(self.positives)
        self.positives = [p for p in self.positives
                          if not (p.kind == KIND_TEXT and p.source == ORIGIN_PHRASE and p.label == phrase)]
        return len(self.positives) != before

    def phrases(self):
        """The phrases added with `add_phrase`, in order."""
        return [p.label for p in self.positives if p.kind == KIND_TEXT and p.source == ORIGIN_PHRASE]

    def replace_group(self, negative, key, prototypes):
        """Put `prototypes` where the examples with this group key were.

        Adding one image's annotations again replaces its earlier examples rather
        than doubling them, and keeps the row where it was in the list.

        Args:
            negative (bool): Which list the group lives in.
            key (tuple): A `Prototype.group_key`.
            prototypes (list[Prototype]): The replacements; may be empty.
        """
        rows = self.negatives if negative else self.positives
        at = next((i for i, p in enumerate(rows) if p.group_key == key), len(rows))
        kept = [p for p in rows if p.group_key != key]
        at = min(at, len(kept))
        rows[:] = kept[:at] + list(prototypes) + kept[at:]

    def merge(self, other):
        """Append another session's examples, skipping ones already here.

        An example is already here if it has the same uid -- a session sent from
        the tool twice -- or the same kind, label and embedding, as a file loaded
        twice has. The other session's threshold and image size are not taken.

        Returns:
            int: How many examples were added.
        """
        added = 0
        for mine, theirs in ((self.positives, other.positives), (self.negatives, other.negatives)):
            for prototype in theirs:
                if any(_same_example(prototype, existing) for existing in mine):
                    continue
                mine.append(prototype.copy())
                added += 1
        if self.model_stem is None:
            self.model_stem = other.model_stem
        return added

    def find(self, uid):
        """The example with this uid, or None."""
        for prototype in self.positives + self.negatives:
            if prototype.uid == uid:
                return prototype
        return None

    def set_enabled(self, uid, enabled):
        """Switch one example on or off without losing it.

        Returns:
            bool: True if the example exists.
        """
        prototype = self.find(uid)
        if prototype is None:
            return False
        prototype.enabled = bool(enabled)
        return True

    def remove(self, uid):
        """Delete one example.

        Returns:
            bool: True if something was removed.
        """
        before = len(self.positives) + len(self.negatives)
        self.positives = [p for p in self.positives if p.uid != uid]
        self.negatives = [p for p in self.negatives if p.uid != uid]
        return len(self.positives) + len(self.negatives) != before

    def clear(self):
        """Forget every example. The model stem and threshold stay."""
        self.positives = []
        self.negatives = []

    def copy(self):
        """An independent copy, so later edits in the tool do not reach a Generator run."""
        session = PromptSession(self.model_stem)
        session.positives = [p.copy() for p in self.positives]
        session.negatives = [p.copy() for p in self.negatives]
        session.confidence = self.confidence
        session.imgsz = self.imgsz
        return session

    # --- reading --------------------------------------------------------------------------------------------------

    def enabled_positives(self):
        return [p for p in self.positives if p.enabled]

    def enabled_negatives(self):
        return [p for p in self.negatives if p.enabled]

    def is_empty(self):
        return not self.positives and not self.negatives

    def has_positives(self):
        """Whether there is anything to predict from."""
        return bool(self.enabled_positives())

    def text_phrase(self):
        """The tool's phrase, if the session holds one."""
        for prototype in self.positives:
            if _is_tool_phrase(prototype):
                return prototype.label
        return None

    def off_size(self, imgsz, enabled_only=True):
        """Visual examples embedded at an image size other than `imgsz`.

        Examples of unknown size are not counted: nothing says they differ.
        """
        rows = self.enabled_positives() + self.enabled_negatives() if enabled_only \
            else self.positives + self.negatives
        return [p for p in rows if p.is_visual and p.imgsz and imgsz and int(p.imgsz) != int(imgsz)]

    def modalities(self):
        """{"text"}, {"visual"}, both, or neither -- over enabled positives."""
        kinds = set()
        for prototype in self.enabled_positives():
            kinds.add("visual" if prototype.is_visual else "text")
        return kinds

    def is_mixed(self):
        """True when text and visual positives share one threshold.

        They are not calibrated alike -- a phrase scored an object 0.95 where an
        example crop of it scored 0.34 -- so a threshold tuned for one hides the
        other's detections.
        """
        return self.modalities() == {"text", "visual"}

    def decoys_can_compete(self):
        return decoys_can_compete(self.enabled_positives())

    def active_negatives(self):
        """The negatives that will actually be sent to the model."""
        return self.enabled_negatives() if self.decoys_can_compete() else []

    def build_classes(self):
        """Class names, class embeddings and the positive count for this session.

        Returns:
            tuple[list[str], torch.Tensor, int]: As `stack_classes`.
        """
        return stack_classes([p.embedding for p in self.enabled_positives()],
                             [p.embedding for p in self.active_negatives()])

    def summary(self):
        """A one-line description, e.g. "2 positive (1 text, 1 visual), 1 negative"."""
        enabled = self.enabled_positives()
        text = sum(1 for p in enabled if not p.is_visual)
        visual = len(enabled) - text

        parts = []
        if text:
            parts.append(f"{text} text")
        if visual:
            parts.append(f"{visual} visual")
        detail = f" ({', '.join(parts)})" if parts else ""

        negatives = len(self.enabled_negatives())
        line = f"{len(enabled)} positive{'' if len(enabled) == 1 else 's'}{detail}"
        if negatives:
            line += f", {negatives} negative{'' if negatives == 1 else 's'}"
        return line

    def check_stem(self, stem):
        """Refuse a model these embeddings were not made with.

        Raises:
            ValueError: If both stems are known and differ.
        """
        if self.model_stem and stem and self.model_stem != stem:
            raise ValueError(
                f"This session was built with '{self.model_stem}' and cannot be used with "
                f"'{stem}': prompt embeddings only mean something to the model that made them."
            )

    # --- persistence ----------------------------------------------------------------------------------------------

    def to_npz(self, path):
        """Write the session to a NPZ file.

        `embeddings`, `names` and `model` are written in the shape
        `YOLOE.save_prompt_embeddings` uses, so the Generator's existing VPE loader
        reads a session file as its enabled positives. The rest of the session rides
        in `session_*` keys. Note that ultralytics' own `load_prompt_embeddings`
        will refuse the file: it requires those three keys and nothing else.

        Returns:
            str: The path written.

        Raises:
            ValueError: If there is nothing to save.
        """
        if not self.positives:
            raise ValueError("A session needs at least one positive example to be saved.")

        dim = self.positives[0].embedding.shape[-1]
        enabled = self.enabled_positives()

        def block(prototypes):
            if not prototypes:
                return np.zeros((0, dim), dtype=np.float32)
            return torch.stack([p.embedding for p in prototypes]).numpy().astype(np.float32)

        def sizes(prototypes):
            return np.asarray([int(p.imgsz) if p.imgsz else -1 for p in prototypes], dtype=np.int64)

        def origins(prototypes):
            # JSON strings, so the file still reads with allow_pickle=False.
            return np.asarray([json.dumps(p.origin or {}) for p in prototypes], dtype=np.str_)

        np.savez_compressed(
            path,
            embeddings=block(enabled)[None, ...],
            names=np.asarray([p.label for p in enabled], dtype=np.str_),
            model=np.asarray(self.model_stem or "", dtype=np.str_),
            session_format=np.asarray(SESSION_FORMAT, dtype=np.str_),
            session_positive_embeddings=block(self.positives),
            session_positive_kinds=np.asarray([p.kind for p in self.positives], dtype=np.str_),
            session_positive_labels=np.asarray([p.label for p in self.positives], dtype=np.str_),
            session_positive_enabled=np.asarray([p.enabled for p in self.positives], dtype=bool),
            session_positive_imgsz=sizes(self.positives),
            session_positive_origins=origins(self.positives),
            session_negative_embeddings=block(self.negatives),
            session_negative_labels=np.asarray([p.label for p in self.negatives], dtype=np.str_),
            session_negative_enabled=np.asarray([p.enabled for p in self.negatives], dtype=bool),
            session_negative_imgsz=sizes(self.negatives),
            session_negative_origins=origins(self.negatives),
            session_confidence=np.asarray(np.nan if self.confidence is None else self.confidence,
                                          dtype=np.float32),
            session_imgsz=np.asarray(-1 if self.imgsz is None else int(self.imgsz), dtype=np.int64),
        )
        return path if str(path).endswith(".npz") else f"{path}.npz"

    @classmethod
    def from_npz(cls, path, expected_stem=None):
        """Read a session file, or a plain prompt-embedding file as positives.

        Read with `allow_pickle=False`, so it cannot execute anything. Legacy `.pt`
        VPE collections are refused: loading them meant unpickling.

        Args:
            path (str): The .npz file.
            expected_stem (str, optional): Refuse the file unless it was made with
                this checkpoint.

        Raises:
            ValueError: If the file is malformed, a `.pt` file, or belongs to another model.
        """
        if str(path).lower().endswith(".pt"):
            raise ValueError("Legacy .pt VPE files are no longer supported. "
                             "Prompts are saved and loaded as .npz files.")
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            if {"embeddings", "model"} - files:
                raise ValueError("Not a prompt session or prompt embedding file.")

            model = data["model"]
            stem = str(model.item()) if model.ndim == 0 else str(model)
            session = cls(stem or None)
            session.check_stem(expected_stem)

            if "session_format" in files:
                cls._read_session_keys(session, data)
            else:
                cls._read_plain_embeddings(session, data)

        return session

    @staticmethod
    def _checked(array, what):
        if array.ndim != 2:
            raise ValueError(f"{what} must be a (count, dimensions) block.")
        if not np.isfinite(array).all():
            raise ValueError(f"{what} contain non-finite values.")
        return torch.from_numpy(array.astype(np.float32))

    @staticmethod
    def _row_details(data, prefix, count, kinds, session_imgsz):
        """Per-example image sizes and origins, or what files without them imply.

        Files from before these were recorded came from the tool, embedded at the
        session's one image size.
        """
        if f"{prefix}_imgsz" in data.files:
            sizes = [int(s) if int(s) > 0 else None for s in data[f"{prefix}_imgsz"]]
        else:
            sizes = [session_imgsz if kind in VISUAL_KINDS else None for kind in kinds]

        if f"{prefix}_origins" in data.files:
            origins = []
            for text in data[f"{prefix}_origins"]:
                try:
                    origin = json.loads(str(text))
                except ValueError:
                    origin = None
                origins.append(origin if isinstance(origin, dict) and origin else None)
        else:
            origins = [None] * count

        if not (len(sizes) == len(origins) == count):
            raise ValueError("Session example details are inconsistent.")
        return sizes, origins

    @classmethod
    def _read_session_keys(cls, session, data):
        session_imgsz = None
        if "session_imgsz" in data.files:
            imgsz = int(data["session_imgsz"])
            session_imgsz = imgsz if imgsz > 0 else None
        session.imgsz = session_imgsz

        positives = cls._checked(data["session_positive_embeddings"], "Positive embeddings")
        kinds = [str(k) for k in data["session_positive_kinds"]]
        labels = [str(label) for label in data["session_positive_labels"]]
        enabled = [bool(e) for e in data["session_positive_enabled"]]
        if not (len(kinds) == len(labels) == len(enabled) == positives.shape[0]):
            raise ValueError("Session positives are inconsistent.")
        sizes, origins = cls._row_details(data, "session_positive", len(kinds), kinds, session_imgsz)

        for vector, kind, label, on, imgsz, origin in zip(positives, kinds, labels, enabled, sizes, origins):
            session.add_positive(kind, vector, label, imgsz=imgsz, origin=origin).enabled = on

        negatives = cls._checked(data["session_negative_embeddings"], "Negative embeddings")
        labels = [str(label) for label in data["session_negative_labels"]]
        enabled = [bool(e) for e in data["session_negative_enabled"]]
        if not (len(labels) == len(enabled) == negatives.shape[0]):
            raise ValueError("Session negatives are inconsistent.")
        sizes, origins = cls._row_details(data, "session_negative", len(labels),
                                          [KIND_DETECTION] * len(labels), session_imgsz)

        for vector, label, on, imgsz, origin in zip(negatives, labels, enabled, sizes, origins):
            session.add_negative(vector, label, imgsz=imgsz, origin=origin).enabled = on

        confidence = float(data["session_confidence"])
        session.confidence = None if np.isnan(confidence) else confidence

    @classmethod
    def _read_plain_embeddings(cls, session, data):
        embeddings = data["embeddings"]
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prompt embeddings must have shape (1, classes, dimensions).")
        vectors = cls._checked(embeddings[0], "Prompt embeddings")
        names = [str(n) for n in data["names"]] if "names" in data.files else []
        for i, vector in enumerate(vectors):
            label = names[i] if i < len(names) else f"imported {i + 1}"
            session.add_positive(KIND_IMPORTED, vector, label, origin={"source": ORIGIN_FILE})
