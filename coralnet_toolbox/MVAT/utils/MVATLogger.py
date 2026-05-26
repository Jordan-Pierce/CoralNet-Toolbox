"""Shared logging helpers for MVAT visibility timing and status output."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


LOGGER_NAME = "coralnet_toolbox.MVAT.visibility"
SECTION_WIDTH = 50


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
    logger.info("")
    logger.info("=" * SECTION_WIDTH)
    logger.info(title)
    logger.info("=" * SECTION_WIDTH)


def log_cam_stage(cam: str, stage: str, elapsed: float, logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.info(f"   {cam}: {elapsed:.4f}s | {stage}: {elapsed:.3f}s")


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
    logger.info(
        f"      {cam}: {cam_time:.4f}s | Prep: {prep:.3f} | Render: {render:.3f} | "
        f"Snap: {screenshot:.3f} | Decode: {decode:.3f} | Depth: {depth:.3f} | "
        f"Finalize: {finalize:.3f} | Residual: {residual:.3f}"
    )


def log_cam_complete(cam: str, elapsed: float, logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.info(f"   ✅ {cam} completed in {elapsed:.2f}s")


def log_summary(title: str, lines: Sequence[str], logger: logging.Logger | None = None) -> None:
    logger = logger or get_visibility_logger()
    logger.info("")
    logger.info(f"SUMMARY: {title}")
    for line in lines:
        logger.info(line)
    logger.info("=" * SECTION_WIDTH)


_visibility_logger = configure_visibility_logging()
