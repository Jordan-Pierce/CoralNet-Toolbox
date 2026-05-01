"""Pure helpers for confidence display and gallery sorting."""

from __future__ import annotations


def confidence_value(annotation) -> float:
    """Return the confidence used for display/sorting.

    Verified annotations use user confidence. Unverified annotations use the
    first machine confidence value.
    """
    confidence_source = annotation.user_confidence if annotation.verified else annotation.machine_confidence

    if confidence_source:
        try:
            return float(next(iter(confidence_source.values())))
        except (TypeError, ValueError, StopIteration):
            return 0.0

    return 0.0


def confidence_bucket_start(confidence) -> int:
    """Map confidence to a 10% bucket start (0, 10, ..., 90)."""
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))
    bucket_start = int(confidence * 10) * 10
    return min(bucket_start, 90)


def confidence_bucket_label(annotation) -> str:
    """Return the display label for a confidence bucket."""
    if annotation.verified:
        return "Verified"

    bucket_start = confidence_bucket_start(confidence_value(annotation))
    if bucket_start >= 90:
        return "90-100%"
    return f"{bucket_start}-{bucket_start + 9}%"


def confidence_bucket_sort_key(annotation):
    """Return a sort key that places numeric buckets before Verified."""
    if annotation.verified:
        return (1, 0, 0.0)

    confidence = max(0.0, min(1.0, confidence_value(annotation)))
    bucket_start = confidence_bucket_start(confidence)
    return (0, bucket_start, confidence)
