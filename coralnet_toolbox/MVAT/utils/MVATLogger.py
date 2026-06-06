"""Shared logging helpers for MVAT visibility timing and status output."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


LOGGER_NAME = "coralnet_toolbox.MVAT.visibility"
SECTION_WIDTH = 50


def _fmt_seconds(value: float) -> str:
    return f"{value:.4g}s"


def configure_visibility_logging(level: int = logging.INFO, stream=None) -> logging.Logger:
    """Configure the dedicated MVAT visibility logger once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_visibility_logger() -> logging.Logger:
    """Return the shared MVAT visibility logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return configure_visibility_logging()
    return logger


def cam_label(index: int, prefix: str = "cam") -> str:
    return f"{prefix} {index}"


def build_camera_labels(camera_paths: Iterable[Any], start_index: int = 1) -> dict[Any, str]:
    return {path: cam_label(index) for index, path in enumerate(camera_paths, start=start_index)}


def label_for_path(path: Any, camera_labels: Mapping[Any, str] | None, fallback: str = "cam") -> str:
    if camera_labels is None:
        return fallback
    return camera_labels.get(path, fallback)


def log_section(title: str, logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.debug("")
    logger.debug("=" * SECTION_WIDTH)
    logger.debug(title)
    logger.debug("=" * SECTION_WIDTH)


def log_cam_stage(cam: str, stage: str, elapsed: float, logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.debug(f"   {cam}: {_fmt_seconds(elapsed)} | {stage}: {_fmt_seconds(elapsed)}")


def log_cam_breakdown(
    cam: str,
    cam_time: float,
    prep: float,
    render: float,
    screenshot: float,
    decode: float,
    depth: float,
    finalize: float,
    residual: float,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or get_visibility_logger()
    logger.debug(
        f"      {cam}: {_fmt_seconds(cam_time)} | Prep: {_fmt_seconds(prep)} | Render: {_fmt_seconds(render)} | "
        f"Snap: {_fmt_seconds(screenshot)} | Decode: {_fmt_seconds(decode)} | Depth: {_fmt_seconds(depth)} | "
        f"Finalize: {_fmt_seconds(finalize)} | Residual: {_fmt_seconds(residual)}"
    )


def log_cam_complete(cam: str, elapsed: float, logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.debug(f"   ✅ {cam} completed in {_fmt_seconds(elapsed)}")


def log_summary(title: str, lines: Sequence[str], logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.debug("")
    logger.debug(f"SUMMARY: {title}")
    for line in lines:
        logger.debug(line)
    logger.debug("=" * SECTION_WIDTH)


_visibility_logger = configure_visibility_logging()
