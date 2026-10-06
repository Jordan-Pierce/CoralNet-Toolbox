"""Where the toolbox keeps its own files.

Everything the toolbox writes for itself lives under one folder in the user's
home directory, on every platform:

    ~/.coralnet-toolbox/
        weights/        stock model checkpoints
        layout/         saved dock layouts
        screenshots/    Capture View's default output folder
        cache/          derived data, safe to delete
            features/           per-image dense feature maps
            embedding/          the Explorer's feature database and indexes
            in_place_training/  generated dataset scaffolding
            active_learning/    Active Learning round runs

Before this, most of it went into the working directory (`.cache/`, and the
weights themselves), so it scattered across every folder the toolbox was
launched from, and per-image features were written beside the images, which
fails on read-only shares.

Set CORALNET_TOOLBOX_HOME to put the whole tree somewhere else: a data drive
with more room than the system drive, or a local disk on a machine whose home
directory roams.

Ultralytics still fetches a few auxiliary files on its own (the AMP check
model during training, the MobileCLIP text encoder for YOLOE text prompts),
and those still land in the working directory.
"""

import hashlib
import os
from pathlib import Path

HOME_ENV = "CORALNET_TOOLBOX_HOME"
WEIGHT_SUFFIXES = (".pt", ".pth")


def app_home():
    """The toolbox's folder, created on first use.

    Returns:
        Path: CORALNET_TOOLBOX_HOME if set, otherwise ~/.coralnet-toolbox.

    Raises:
        RuntimeError: If the folder cannot be created, naming it and the
            variable that moves it.
    """
    configured = os.environ.get(HOME_ENV)
    path = Path(configured).expanduser() if configured else Path.home() / ".coralnet-toolbox"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"The toolbox cannot write to {path} ({e}). "
                           f"Set {HOME_ENV} to a folder it can write to.") from e
    return path


def app_dir(*parts):
    """A folder under app_home(), created on first use.

    Args:
        *parts (str): Path components below the home folder.

    Returns:
        Path: The folder.
    """
    path = app_home().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts):
    """A folder under app_home()/cache, for derived data that is safe to delete.

    Args:
        *parts (str): Path components below the cache folder.

    Returns:
        Path: The folder.
    """
    return app_dir("cache", *parts)


def weights_dir():
    """The folder stock model checkpoints are kept in.

    Returns:
        Path: app_home()/weights.
    """
    return app_dir("weights")


def feature_cache_path(image_path):
    """Where an image's dense feature map is cached.

    The file name keeps the image's base name, so the folder stays readable,
    and adds a hash of its full path: images from different folders often
    share a name (frame_0001, IMG_0001), and would otherwise overwrite each
    other's features in this one shared folder.

    Args:
        image_path (str): The image's path.

    Returns:
        str: Path to the .npy file, which need not exist yet.
    """
    key = os.path.normcase(os.path.abspath(image_path)).replace("\\", "/")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return (cache_dir("features") / f"{stem}_{digest}_features.npy").as_posix()


def find_weights(name):
    """An existing copy of a stock checkpoint, or None.

    The working directory is checked first: that is where earlier versions
    downloaded to, and a 3.5 GB SAM 3 checkpoint should not be fetched twice.

    Args:
        name (str): Checkpoint file name, e.g. "sam3.pt".

    Returns:
        str | None: Path to the checkpoint, or None if neither place has it.
    """
    if os.path.isfile(name):
        return name
    candidate = weights_dir() / name
    return candidate.as_posix() if candidate.is_file() else None


def resolve_weights(model_path):
    """Where ultralytics should load, or download, a checkpoint.

    Only a bare stock name ("yolo11n.pt") is redirected, to an existing copy
    or else into the weights folder, where ultralytics then downloads it. A
    path with a directory in it, a model config (.yaml), or anything else
    passes through unchanged.

    Args:
        model_path (str | Path): Checkpoint name or path.

    Returns:
        str: The path to hand to ultralytics.
    """
    model_path = str(model_path)
    if os.path.dirname(model_path) or not model_path.lower().endswith(WEIGHT_SUFFIXES):
        return model_path
    return find_weights(model_path) or (weights_dir() / model_path).as_posix()
