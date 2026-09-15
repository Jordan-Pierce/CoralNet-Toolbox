import os

import numpy as np
import torch

from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


# The smallest prompt-free checkpoint. Only its `names` are ever read: the
# 4,585-entry vocabulary YOLOE's prompt-free variants are trained against. The
# weights themselves are thrown away, and once the cache below is written the
# file is never needed again.
VOCABULARY_CHECKPOINT = "yoloe-11s-seg-pf.pt"
VOCABULARY_CHECKPOINT_MB = 27

# Written next to the weights, which is where ultralytics already downloads.
VOCABULARY_CACHE_SUFFIX = "-vocabulary.npz"

# Encoding chunk. Big enough that the per-call overhead disappears, small
# enough that a 4,585-name run can report progress and stay interruptible.
ENCODE_CHUNK = 256


# ----------------------------------------------------------------------------------------------------------------------
# Model introspection
# ----------------------------------------------------------------------------------------------------------------------


def inner_model(model):
    """The `YOLOEModel` inside a `YOLOE` wrapper, or None.

    Args:
        model: A loaded `YOLOE`, or anything shaped like one.

    Returns:
        The inner model, or None if there is nothing usable.
    """
    return getattr(model, 'model', None)


def detection_head(model):
    """The `YOLOEDetect` head, or None if the model is not shaped as expected."""
    inner = inner_model(model)
    layers = getattr(inner, 'model', None)
    try:
        return layers[-1]
    except (TypeError, IndexError, KeyError):
        return None


def promptable(model):
    """Whether this model can still turn phrases into class embeddings.

    Fusing the head (`get_vocab`/`set_vocab`) swaps `reprta` for `nn.Identity`
    and cannot be undone, which would leave `get_text_pe` returning vectors from
    a different space than the VPEs they would be compared against. Nothing in
    the toolbox fuses, so this is a guard rather than a branch anyone reaches.

    Args:
        model: A loaded `YOLOE`.

    Returns:
        bool: True if text prompt embeddings from this model are meaningful.
    """
    if model is None:
        return False
    head = detection_head(model)
    if head is None:
        return False
    if getattr(head, 'is_fused', False):
        return False
    return hasattr(inner_model(model), 'get_text_pe')


def to_model_space(model, tensor):
    """Move a tensor onto the model's own device and dtype.

    Everything in this module works in CPU float32 so that ranking is exact and
    testable, but anything handed back to `set_classes` has to match the weights
    it will be multiplied against -- which on this project's usual setup means
    CUDA, and half precision whenever quantization is on.

    Args:
        model: A loaded `YOLOE`.
        tensor (torch.Tensor): Any tensor.

    Returns:
        torch.Tensor: Cast to the model's device and dtype, or unchanged if the
        model has no parameters to read them from.
    """
    try:
        param = next(inner_model(model).parameters())
    except (AttributeError, TypeError, StopIteration):
        return tensor
    return tensor.to(param.device, param.dtype)


def checkpoint_stem(model):
    """The checkpoint identifier ultralytics binds prompt embeddings to, or None.

    `_prompt_embedding_model` is private, so a missing or failing method means
    the cache simply goes unkeyed rather than that anything breaks.
    """
    if model is None:
        return None
    try:
        return model._prompt_embedding_model()
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Visual prompt embeddings
# ----------------------------------------------------------------------------------------------------------------------


def usable_vp_predictor(predictor):
    """Whether this predictor can take prompts and hand back a VPE.

    Tested by capability rather than class: these two methods are all VPE
    extraction needs, and a plain `SegmentationPredictor` -- what a prompt-free
    `predict()` leaves behind -- has neither.
    """
    return hasattr(predictor, 'set_prompts') and hasattr(predictor, 'get_vpe')


def _canonical_quantize(value):
    """8/16/32 for any spelling ultralytics accepts, None when unset."""
    if value is None:
        return None
    text = str(value).lower()
    for bits, spellings in ((8, {"8", "int8", "w8a8"}),
                            (16, {"16", "fp16", "w16a16"}),
                            (32, {"32", "fp32", "w32a32"})):
        if text in spellings:
            return bits
    return text


def _predictor_matches(predictor, overrides):
    """Whether `Model.predict` would keep this predictor for a call with `overrides`.

    `Model.predict` rebuilds the predictor whenever `args.device` differs (8.4.129
    and 8.4.148), or `args.quantize` does (8.4.148) -- and it rebuilds it as the
    task's default predictor, not the visual-prompt one it replaces.
    """
    args = getattr(predictor, 'args', None)
    if args is None:
        return False
    if 'device' in overrides and str(getattr(args, 'device', None)) != str(overrides['device']):
        return False
    if 'quantize' in overrides and (_canonical_quantize(getattr(args, 'quantize', None))
                                    != _canonical_quantize(overrides['quantize'])):
        return False
    return True


def ensure_vp_predictor(model, **predict_args):
    """Make sure `model.predictor` is a visual-prompt predictor.

    With `predict_args` (imgsz, device, quantize) the warm-up always runs with
    them, even over a usable predictor. `get_vpe` letterboxes at the predictor's
    own `args.imgsz`, which is whatever the last predict call used -- and the
    warm-up below uses 640. An embedding extracted at 640 is not the embedding an
    in-image visual prompt produces at 1024, so a caller who needs the two to
    match has to say which size it predicts at.

    A predictor built with another device or precision is discarded first.
    `YOLOE.predict` reuses a visual-prompt predictor of the right class, then hands
    off to `Model.predict`, which compares `args.device` (and on 8.4.148
    `args.quantize`) with `!=` and on any difference replaces the predictor with
    the task's *default* one -- a plain `SegmentationPredictor`, with no `get_vpe`.
    `reload_model` warms up with no device, so its predictor had `device=None`;
    asking for "cuda:0" then silently swapped it out, and "Generate VPEs" failed
    with "no visual-prompt predictor available". Starting from no predictor lets
    `YOLOE.predict` build one with the requested device and precision, which
    `Model.predict` then keeps. One clean retry covers any other rebuild trigger a
    later ultralytics adds.

    From ultralytics 8.4.148, `YOLOE.set_classes` ends with
    `self.predictor = None`; 8.4.129 only refreshed the predictor's class names.
    So any code that reaches for `predictor.set_prompts` or `predictor.get_vpe`
    after a VPE has been applied finds nothing there -- which is what made a
    second "Generate VPEs" fail with
    "'NoneType' object has no attribute 'set_prompts'".

    Running the one-box warm-up rebuilds it. That resets the head to a single
    class, which is irrelevant here: VPE extraction uses the encoder, not the
    class embeddings.

    Args:
        model: A loaded `YOLOE`, or None.

    Returns:
        bool: True if a visual-prompt predictor is now in place.
    """
    if model is None:
        return False

    overrides = {k: v for k, v in predict_args.items() if v is not None}
    predictor = getattr(model, 'predictor', None)
    if not overrides and usable_vp_predictor(predictor):
        return True

    if predictor is not None and not _predictor_matches(predictor, overrides):
        model.predictor = None

    warmup = dict(imgsz=640, conf=0.99, verbose=False)
    warmup.update(overrides)

    for _attempt in range(2):
        model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            visual_prompts=dict(bboxes=np.array([[120, 425, 160, 445]]),
                                cls=np.zeros(1)),
            predictor=YOLOEVPSegPredictor,
            **warmup,
        )
        if usable_vp_predictor(getattr(model, 'predictor', None)):
            return True
        model.predictor = None

    return False


def vpe_from_prompts(model, source, prompts, **predict_args):
    """Extract one normalized VPE for `prompts` drawn on `source`.

    Args:
        model: A loaded `YOLOE`.
        source: Anything `YOLOEVPDetectPredictor.get_vpe` accepts as a single
            image -- a path, or the BGR array of a work-area crop.
        prompts (dict): `bboxes`/`masks` plus `cls`, in `source` pixels.
        **predict_args: imgsz/device/quantize to extract at; see
            `ensure_vp_predictor` for why imgsz matters.

    Returns:
        torch.Tensor | None: Shape (1, N, D), L2-normalized, or None if no
        visual-prompt predictor could be built.
    """
    if not ensure_vp_predictor(model, **predict_args):
        return None

    model.predictor.set_prompts(prompts)
    vpe = model.predictor.get_vpe(source)
    return torch.nn.functional.normalize(vpe, p=2, dim=-1)


def as_matrix(vpes):
    """Flatten assorted VPE tensors into one normalized (K, D) float32 matrix.

    Accepts the (1, N, D) tensors `get_vpe` returns as well as bare (D,)
    vectors, since imported NPZ prototypes arrive per-class.

    Args:
        vpes: Iterable of tensors.

    Returns:
        torch.Tensor: Shape (K, D) on the CPU, every row unit length. Empty
        input gives an empty tensor rather than an error.
    """
    rows = []
    for vpe in vpes:
        if vpe is None:
            continue
        flat = vpe.detach().to('cpu', dtype=torch.float32).reshape(-1, vpe.shape[-1])
        rows.append(flat)

    if not rows:
        return torch.empty((0, 0))

    matrix = torch.cat(rows, dim=0)
    return torch.nn.functional.normalize(matrix, p=2, dim=-1)


# ----------------------------------------------------------------------------------------------------------------------
# Text prompt embeddings
# ----------------------------------------------------------------------------------------------------------------------


class TextEmbedder:
    """Encodes phrases with one model's text head, remembering what it encoded.

    Two costs are worth avoiding and this exists to avoid both. `YOLOE.get_text_pe`
    leaves `cache_clip_model` at its default of False, so every call rebuilds
    the MobileCLIP text encoder from the 572 MB TorchScript file -- measured at
    1.51 s cold and 0.40 s warm per call, against 0.08 s when the encoder is
    kept. And the words a user ranks are the same words every time they press
    the button, so the phrase cache turns repeat rankings into a dot product.
    """

    def __init__(self, model):
        """
        Args:
            model: A loaded `YOLOE`.
        """
        self.model = model
        self._cache = {}

    def clear(self):
        """Forget cached phrase embeddings (the CLIP encoder stays on the model)."""
        self._cache.clear()

    def encode(self, texts, progress=None):
        """Embed phrases into the model's prompt space.

        Args:
            texts (list[str]): Phrases to encode. Order is preserved and
                duplicates are collapsed before hitting the encoder.
            progress (callable, optional): Called as `progress(done, total)`
                after each chunk, counting only phrases that were not cached.

        Returns:
            torch.Tensor: Shape (len(texts), D), L2-normalized, CPU float32.

        Raises:
            RuntimeError: If this model cannot produce text embeddings.
        """
        if not promptable(self.model):
            raise RuntimeError(
                "This model cannot produce text embeddings: its head has been fused."
            )

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        total = len(missing)
        done = 0

        for start in range(0, total, ENCODE_CHUNK):
            chunk = missing[start:start + ENCODE_CHUNK]
            # cache_clip_model=True keeps the TorchScript encoder on the model
            # between chunks; without it every chunk pays the rebuild.
            embeddings = inner_model(self.model).get_text_pe(chunk, cache_clip_model=True)
            embeddings = embeddings.detach().to('cpu', dtype=torch.float32).reshape(len(chunk), -1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            for text, row in zip(chunk, embeddings):
                self._cache[text] = row
            done += len(chunk)
            if progress is not None:
                progress(done, total)

        if not texts:
            return torch.empty((0, 0))
        return torch.stack([self._cache[t] for t in texts])

    def prime(self, texts, embeddings):
        """Seed the cache with embeddings read back from disk.

        Args:
            texts (list[str]): Phrases, in the same order as `embeddings`.
            embeddings (torch.Tensor): Shape (len(texts), D).
        """
        normalized = torch.nn.functional.normalize(
            embeddings.detach().to('cpu', dtype=torch.float32), p=2, dim=-1)
        for text, row in zip(texts, normalized):
            self._cache[text] = row


# ----------------------------------------------------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------------------------------------------------


def rank(vpe_matrix, text_matrix, texts, top_k=None):
    """Rank phrases by how well they align with a set of visual prompts.

    Args:
        vpe_matrix (torch.Tensor): Shape (K, D), one row per reference.
        text_matrix (torch.Tensor): Shape (T, D), one row per phrase.
        texts (list[str]): The phrases, in `text_matrix` row order.
        top_k (int, optional): Keep only this many. None keeps everything.

    Returns:
        list[dict]: Best first, each with:
            `text`, `score` (against the mean of the references), `lowest` and
            `highest` (the worst and best single reference), and `spread`
            (their difference -- large means the references disagree about this
            phrase, which usually means the references are mixed).
    """
    if vpe_matrix.numel() == 0 or text_matrix.numel() == 0 or not texts:
        return []

    # The mean of unit vectors is not a unit vector; renormalizing keeps the
    # headline score on the same 0-1 scale as the per-reference ones.
    centroid = torch.nn.functional.normalize(vpe_matrix.mean(dim=0, keepdim=True), p=2, dim=-1)

    per_reference = vpe_matrix @ text_matrix.T          # (K, T)
    against_mean = (centroid @ text_matrix.T)[0]        # (T,)

    ranked = []
    for index, text in enumerate(texts):
        column = per_reference[:, index]
        ranked.append({
            'text': text,
            'score': float(against_mean[index]),
            'lowest': float(column.min()),
            'highest': float(column.max()),
            'spread': float(column.max() - column.min()),
        })

    ranked.sort(key=lambda row: row['score'], reverse=True)
    return ranked if top_k is None else ranked[:top_k]


def margin(ranked):
    """How far the winner is clear of the runner-up.

    Absolute scores are not calibrated -- a correct match sits anywhere from
    roughly 0.40 to 0.60 -- so a fixed threshold means nothing and this does.
    A wide margin is a phrase worth typing; a narrow one means the vocabulary
    holds several equally good handles and any of them would serve.

    Args:
        ranked (list[dict]): Output of `rank`.

    Returns:
        float: Top score minus runner-up, or 0.0 if there is no runner-up.
    """
    if len(ranked) < 2:
        return 0.0
    return ranked[0]['score'] - ranked[1]['score']


# ----------------------------------------------------------------------------------------------------------------------
# The prompt-free vocabulary
# ----------------------------------------------------------------------------------------------------------------------


def vocabulary_cache_path(stem, directory=None):
    """Where this checkpoint's encoded vocabulary lives.

    Keyed by checkpoint because `get_tpe` runs every phrase through that
    model's own `reprta` head: the same word encoded by two checkpoints gives
    two unrelated vectors.

    Args:
        stem (str): Checkpoint stem, from `checkpoint_stem`.
        directory (str, optional): Defaults to the working directory, which is
            where ultralytics already downloads weights.

    Returns:
        str: Path to the NPZ, which need not exist yet.
    """
    name = f"{stem or 'yoloe'}{VOCABULARY_CACHE_SUFFIX}"
    return os.path.join(directory or os.getcwd(), name)


def read_vocabulary_cache(path):
    """Read a cached vocabulary, or None if there is not a usable one.

    Read with `allow_pickle=False`: this file is regenerable, so there is no
    reason to let it execute anything.

    Args:
        path (str): NPZ written by `write_vocabulary_cache`.

    Returns:
        tuple[list[str], torch.Tensor] | None: Names and their (T, D) embeddings.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            names = [str(n) for n in data['names']]
            embeddings = torch.from_numpy(np.asarray(data['embeddings'], dtype=np.float32))
    except Exception:
        return None

    if embeddings.ndim != 2 or embeddings.shape[0] != len(names) or not torch.isfinite(embeddings).all():
        return None
    return names, embeddings


def write_vocabulary_cache(path, names, embeddings):
    """Persist an encoded vocabulary so it is never rebuilt.

    Args:
        path (str): Destination NPZ.
        names (list[str]): Phrases.
        embeddings (torch.Tensor): Shape (len(names), D).

    Returns:
        str | None: The path written, or None if it could not be written.
    """
    try:
        np.savez_compressed(
            path,
            names=np.array(names, dtype=np.str_),
            embeddings=embeddings.detach().to('cpu', dtype=torch.float32).numpy(),
        )
        return path
    except Exception:
        return None


def load_vocabulary_names():
    """The 4,585 names YOLOE's prompt-free variants are trained against.

    They ship inside the `-pf` checkpoints as `model.names` and nowhere else --
    no plain list is published -- so the smallest one is downloaded and its
    names read. The weights are discarded; once `write_vocabulary_cache` has
    run, this is never called again.

    Returns:
        list[str]: Vocabulary names.

    Raises:
        RuntimeError: If the checkpoint could not be fetched or read.
    """
    try:
        # `torch_safe_load` is ultralytics' own guarded loader: it downloads the
        # asset if it is missing and returns the raw checkpoint without building
        # the model, which is all that is wanted here -- the weights are thrown
        # away and only `names` is kept.
        from ultralytics.nn.tasks import torch_safe_load

        checkpoint, _ = torch_safe_load(VOCABULARY_CHECKPOINT)
        inner = checkpoint.get('model') if isinstance(checkpoint, dict) else None
        names = getattr(inner, 'names', None)
    except Exception as e:
        raise RuntimeError(f"Could not read the prompt-free vocabulary: {e}") from e

    names = list(names.values()) if isinstance(names, dict) else list(names or [])
    if not names:
        raise RuntimeError("The prompt-free checkpoint carried no vocabulary.")
    return [str(n) for n in names]


def vocabulary_embeddings(embedder, stem=None, directory=None, progress=None):
    """The full vocabulary, encoded for this model, from cache when possible.

    First call downloads the name list and encodes it -- measured at 79 s for
    all 4,585 names on a CPU, far less on a GPU. Every call after that reads a
    9.4 MB NPZ.

    Args:
        embedder (TextEmbedder): Bound to the model the ranking will use.
        stem (str, optional): Checkpoint stem; defaults to the embedder's model.
        directory (str, optional): Where the cache lives.
        progress (callable, optional): `progress(done, total)` while encoding.

    Returns:
        tuple[list[str], torch.Tensor]: Names and their (T, D) embeddings.
    """
    stem = stem or checkpoint_stem(embedder.model)
    path = vocabulary_cache_path(stem, directory)

    cached = read_vocabulary_cache(path)
    if cached is not None:
        names, embeddings = cached
        embedder.prime(names, embeddings)
        return names, embeddings

    names = load_vocabulary_names()
    embeddings = embedder.encode(names, progress=progress)
    write_vocabulary_cache(path, names, embeddings)
    return names, embeddings


def vocabulary_is_cached(stem, directory=None):
    """Whether the vocabulary can be ranked without a download or a long encode."""
    return os.path.exists(vocabulary_cache_path(stem, directory))
