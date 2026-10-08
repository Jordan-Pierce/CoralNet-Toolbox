"""
Annotation filters for the Explorer gallery.

A filter is one thing the gallery can be narrowed by: an annotation's image,
label, type or verified status. Each filter knows how to read its value off an
annotation and what the choices are; FilterState holds what the user picked.
The gallery tests annotations with FilterState.matches and the filter dialog
counts them with FacetIndex, so both agree on what a filter means.

Adding a filter means writing one AnnotationFilter subclass and adding it to
ANNOTATION_FILTERS. The dialog builds a tab for it and the gallery applies it
without further changes.

Rules shared by every filter:
    - Within one filter the picked values are OR'd, across filters they are AND'd.
    - The picked values are exactly what passes. Everything picked is stored
      as no restriction (None), so images or labels added later still pass;
      nothing picked (an empty set) lets nothing through.
"""

from __future__ import annotations

import os
from collections import namedtuple

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# One choice in a filter. `keys` are lowercase strings the dialog searches.
FilterOption = namedtuple("FilterOption", "value text tooltip keys")


# ----------------------------------------------------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------------------------------------------------


class AnnotationFilter:
    """Base class: one thing annotations can be filtered by."""

    key = ""
    title = ""
    noun = ("item", "items")

    def value_of(self, annotation):
        raise NotImplementedError

    def options(self, main_window):
        """Return the choices as a list of FilterOption."""
        raise NotImplementedError

    def fallback_option(self, value, annotation):
        """Option for a value an annotation has but options() did not list."""
        text = str(value)
        return FilterOption(value, text, text, (text.lower(),))

    def name_of(self, value, main_window=None):
        """Display name of one value, for the tooltip."""
        return str(value)

    def accepts(self, annotation, spec):
        return self.value_of(annotation) in spec

    def describe(self, spec, main_window=None, limit=15):
        """One line for the Filter button's tooltip, e.g. 'Labels: Coral, Sand'."""
        if spec is None:
            return f"{self.title}: all"
        if not spec:
            return f"{self.title}: none"
        names = sorted(self.name_of(value, main_window) for value in spec)
        text = ", ".join(names[:limit])
        if len(names) > limit:
            text += f" and {len(names) - limit} more"
        return f"{self.title}: {text}"


class ImageFilter(AnnotationFilter):
    """Keyed by full path, so two images with the same file name stay apart."""

    key = "image"
    title = "Images"
    noun = ("image", "images")

    def value_of(self, annotation):
        return annotation.image_path

    def options(self, main_window):
        image_window = getattr(main_window, "image_window", None)
        raster_manager = getattr(image_window, "raster_manager", None)
        paths = list(getattr(raster_manager, "image_paths", None) or [])
        return [self._option(path) for path in paths]

    def fallback_option(self, value, annotation):
        return self._option(value)

    @staticmethod
    def _option(path):
        name = os.path.basename(path)
        return FilterOption(path, name, path, (name.lower(),))

    def name_of(self, value, main_window=None):
        return os.path.basename(value)


class LabelFilter(AnnotationFilter):
    """Keyed by label id, so renaming a label does not drop it from the filter."""

    key = "label"
    title = "Labels"
    noun = ("label", "labels")

    def value_of(self, annotation):
        return annotation.label.id

    def options(self, main_window):
        label_window = getattr(main_window, "label_window", None)
        labels = getattr(label_window, "labels", None) or []
        return [self._option(label) for label in labels]

    def fallback_option(self, value, annotation):
        return self._option(annotation.label)

    @staticmethod
    def _option(label):
        short = label.short_label_code
        long = label.long_label_code
        return FilterOption(label.id, short, long, (short.lower(), long.lower()))

    def name_of(self, value, main_window=None):
        label_window = getattr(main_window, "label_window", None)
        for label in getattr(label_window, "labels", None) or []:
            if label.id == value:
                return label.short_label_code
        return "(deleted label)"


class TypeFilter(AnnotationFilter):
    key = "type"
    title = "Types"
    noun = ("type", "types")

    TYPES = ("PatchAnnotation", "RectangleAnnotation", "PolygonAnnotation", "MultiPolygonAnnotation")

    def value_of(self, annotation):
        return type(annotation).__name__

    def options(self, main_window):
        return [self.fallback_option(name, None) for name in self.TYPES]

    def fallback_option(self, value, annotation):
        text = self.name_of(value)
        return FilterOption(value, text, value, (text.lower(),))

    def name_of(self, value, main_window=None):
        return value[:-len("Annotation")] if value.endswith("Annotation") else value


class StatusFilter(AnnotationFilter):
    key = "status"
    title = "Status"
    noun = ("status", "statuses")

    def value_of(self, annotation):
        return "Verified" if annotation.verified else "Unverified"

    def options(self, main_window):
        return [self.fallback_option(value, None) for value in ("Verified", "Unverified")]


ANNOTATION_FILTERS = (
    ImageFilter(),
    LabelFilter(),
    TypeFilter(),
    StatusFilter(),
)

FILTERS_BY_KEY = {f.key: f for f in ANNOTATION_FILTERS}


# ----------------------------------------------------------------------------------------------------------------------
# Filter state
# ----------------------------------------------------------------------------------------------------------------------


class FilterState:
    """What the user picked for each filter, as a frozenset of values per key.

    Filters not stored do not restrict; a stored empty set lets nothing through.
    """

    def __init__(self, specs=None):
        self._specs = {}
        for key, spec in (specs or {}).items():
            self.set(key, spec)

    def get(self, key):
        return self._specs.get(key)

    def set(self, key, spec):
        """Store the picked values; None removes the restriction."""
        if spec is None:
            self._specs.pop(key, None)
        else:
            self._specs[key] = frozenset(spec)

    def is_active(self, key):
        return key in self._specs

    def active_count(self):
        return len(self._specs)

    def is_empty(self):
        return not self._specs

    def copy(self):
        return FilterState(dict(self._specs))

    def __eq__(self, other):
        return isinstance(other, FilterState) and self._specs == other._specs

    def __ne__(self, other):
        return not self.__eq__(other)

    def matches(self, annotation):
        """True if the annotation passes every active filter."""
        for key, spec in self._specs.items():
            if not FILTERS_BY_KEY[key].accepts(annotation, spec):
                return False
        return True

    def prune(self, key, available):
        """Drop picked values that no longer exist. Returns True if anything changed.

        If none of the picked values survive the filter clears, matching how the
        old combo boxes fell back to 'All' rather than filtering to nothing.
        """
        spec = self._specs.get(key)
        if spec is None:
            return False
        surviving = spec & frozenset(available)
        if surviving == spec:
            return False
        self.set(key, surviving or None)
        return True

    def describe(self, main_window=None):
        """One line per filter, 'all' where it does not restrict, for a tooltip."""
        return "\n".join(f.describe(self._specs.get(f.key), main_window) for f in ANNOTATION_FILTERS)


# ----------------------------------------------------------------------------------------------------------------------
# Facet counting
# ----------------------------------------------------------------------------------------------------------------------


class FacetCounts:
    """Result of FacetIndex.count.

    Attributes:
        total: annotations passing every filter.
        images: distinct images among them.
        crops: how many of them still need a crop before the gallery can show them.
        per_key: per filter, an array of counts per option of annotations that
            pass every *other* filter.
    """

    def __init__(self, total, images, crops, per_key):
        self.total = total
        self.images = images
        self.crops = crops
        self.per_key = per_key


class FacetIndex:
    """Each annotation's filter values, encoded once so counting is vectorised.

    Built when the filter dialog opens. Values become integer codes into each
    filter's option list: options the project lists come first, then any value
    only an annotation has.
    """

    def __init__(self, annotations, main_window=None, uncroppable=(), filters=ANNOTATION_FILTERS):
        self.annotations = list(annotations)
        self.filters = tuple(filters)
        self.n = len(self.annotations)
        self.options = {}
        self.codes = {}
        self._code_of = {}

        for filt in self.filters:
            options = list(filt.options(main_window))
            code_of = {}
            for i, option in enumerate(options):
                code_of.setdefault(option.value, i)
            codes = np.empty(self.n, dtype=np.int64)
            for row, annotation in enumerate(self.annotations):
                value = filt.value_of(annotation)
                code = code_of.get(value)
                if code is None:
                    code = len(options)
                    code_of[value] = code
                    options.append(filt.fallback_option(value, annotation))
                codes[row] = code
            self.options[filt.key] = options
            self.codes[filt.key] = codes
            self._code_of[filt.key] = code_of

        uncroppable = set(uncroppable or ())
        self.needs_crop = np.fromiter(
            (getattr(a, "cropped_image", None) is None and a.id not in uncroppable
             for a in self.annotations),
            dtype=bool, count=self.n)

    def _mask(self, filt, spec):
        allowed = np.zeros(len(self.options[filt.key]), dtype=bool)
        code_of = self._code_of[filt.key]
        for value in spec:
            code = code_of.get(value)
            if code is not None:
                allowed[code] = True
        return allowed[self.codes[filt.key]]

    def count(self, state):
        """Count matches for `state`, plus faceted counts for every filter.

        A filter's counts use every other filter but not itself, so a label that
        is not selected still shows how many annotations it would add.
        """
        passes = {}
        fails = np.zeros(self.n, dtype=np.int16)
        for filt in self.filters:
            spec = state.get(filt.key)
            if spec is None:
                continue
            passed = self._mask(filt, spec)
            passes[filt.key] = passed
            fails += ~passed

        base = fails == 0
        one_fail = fails == 1

        per_key = {}
        for filt in self.filters:
            passed = passes.get(filt.key)
            # Passing every other filter: everything in `base`, plus anything
            # whose only failure is this filter.
            others = base if passed is None else base | (one_fail & ~passed)
            per_key[filt.key] = np.bincount(self.codes[filt.key][others],
                                            minlength=len(self.options[filt.key]))

        total = int(base.sum())
        image_codes = self.codes.get("image")
        images = int(np.unique(image_codes[base]).size) if image_codes is not None else 0
        crops = int((base & self.needs_crop).sum())
        return FacetCounts(total, images, crops, per_key)
