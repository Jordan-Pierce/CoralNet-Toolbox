import warnings

import copy
import datetime

import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------


# Supported field types. The window maps each of these to a concrete editor widget.
FIELD_TYPES = ('string', 'text', 'bool', 'int', 'float', 'choice', 'multichoice', 'date')

# Per-type fallback used when a field definition omits 'default'.
TYPE_DEFAULTS = {
    'string': '',
    'text': '',
    'bool': False,
    'int': 0,
    'float': 0.0,
    'choice': '',
    'multichoice': [],
    'date': '',
}

# Keys that arrive in annotation.data from an importer and deserve a real typed
# definition rather than one inferred from whatever value happened to be seen
# first. taglab_key names the slot the value came from (and returns to) in a
# TagLab region dict.
SEEDED_FIELDS = {
    'instance_name': {'type': 'string', 'label': 'Instance Name', 'taglab_key': 'instance name',
                      'description': 'TagLab instance name.'},
    'blob_name': {'type': 'string', 'label': 'Blob Name', 'taglab_key': 'blob name',
                  'description': 'TagLab blob name.'},
    'note': {'type': 'text', 'label': 'Note', 'taglab_key': 'note',
             'description': "Free-form note. Exported to TagLab's note field."},
    # taglab_key marks this as owned by TagLab's 'id' slot, which the exporter
    # fills from its own per-file counter (a preserved id could collide with a
    # natively drawn annotation). The mapping keeps it out of the note text.
    'taglab_id': {'type': 'int', 'label': 'TagLab ID', 'taglab_key': 'id',
                  'description': 'Region identifier assigned by TagLab on import.'},
    'Dot': {'type': 'string', 'label': 'Dot', 'description': 'Viscore dot identifier.'},
    'ReprojectionError': {'type': 'float', 'label': 'Reprojection Error',
                          'description': 'Viscore reprojection error.'},
    'ViewIndex': {'type': 'int', 'label': 'View Index', 'description': 'Viscore view index.'},
    'ViewCount': {'type': 'int', 'label': 'View Count', 'description': 'Viscore view count.'},
}

# Imported keys that duplicate something the built-in tier already computes from
# geometry. Promoting them would persist a stale copy of a derived value, so
# they are dropped from annotation.data instead.
DERIVED_IMPORT_KEYS = ('bbox', 'centroid', 'area', 'perimeter', 'class name', 'class_name')


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class _Missing:
    """Sentinel distinguishing 'absent' from a stored None."""
    pass


_MISSING = _Missing()


class MetaDataField:
    """A single user-defined metadata field definition.

    A field is pure data: it knows its type, its default, and how to coerce an
    arbitrary value into that type. It holds no per-annotation state -- values
    live on the annotations themselves, keyed by this field's name.
    """

    def __init__(self,
                 name,
                 type='string',
                 label=None,
                 default=None,
                 options=None,
                 minimum=None,
                 maximum=None,
                 step=None,
                 decimals=2,
                 max_length=0,
                 description='',
                 visible=True,
                 taglab_key=None):
        """Initialize a metadata field definition."""
        if type not in FIELD_TYPES:
            raise ValueError(f"Unknown metadata field type: {type}")

        self.name = str(name).strip()
        if not self.name:
            raise ValueError("Metadata field name cannot be empty.")

        self.type = type
        self.label = (label or self.name).strip()
        self.options = list(options) if options else []
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.decimals = decimals
        self.max_length = max_length
        self.description = description or ''
        self.visible = bool(visible)
        self.taglab_key = taglab_key

        # Resolve the default last: coercion depends on every attribute above.
        if default is None:
            default = TYPE_DEFAULTS[self.type]
            # An unspecified default for a choice field is the first option,
            # which keeps the combo box and the stored value in agreement.
            if self.type == 'choice' and self.options:
                default = self.options[0]
        self.default = self.coerce(default)

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    def coerce(self, value):
        """Convert a value to this field's type, raising ValueError if impossible."""
        if value is None:
            return copy.deepcopy(TYPE_DEFAULTS[self.type])

        if self.type in ('string', 'text'):
            text = str(value)
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length]
            return text

        if self.type == 'bool':
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ('true', 'yes', '1', 't', 'y'):
                    return True
                if lowered in ('false', 'no', '0', 'f', 'n', ''):
                    return False
                raise ValueError(f"Cannot interpret {value!r} as a boolean.")
            return bool(value)

        if self.type == 'int':
            # float() first so '3.0' and 3.7 both survive; int(str) rejects both.
            return self._clamp(int(round(float(value))))

        if self.type == 'float':
            return self._clamp(float(value))

        if self.type == 'choice':
            text = str(value)
            if self.options and text not in self.options:
                raise ValueError(f"{text!r} is not one of {self.options}.")
            return text

        if self.type == 'multichoice':
            if isinstance(value, str):
                values = [part.strip() for part in value.split(',') if part.strip()]
            else:
                values = [str(item) for item in value]
            if self.options:
                unknown = [item for item in values if item not in self.options]
                if unknown:
                    raise ValueError(f"{unknown} are not among {self.options}.")
                # Preserve the schema's option order so equality against the
                # default is order-insensitive and the sparse check stays sound.
                values = [item for item in self.options if item in values]
            return values

        if self.type == 'date':
            if isinstance(value, (datetime.date, datetime.datetime)):
                return value.strftime('%Y-%m-%d')
            text = str(value).strip()
            if not text:
                return ''
            # Validate by parsing; store the canonical ISO form.
            return datetime.datetime.strptime(text[:10], '%Y-%m-%d').strftime('%Y-%m-%d')

        raise ValueError(f"Unhandled field type: {self.type}")

    def _clamp(self, number):
        """Clamp a numeric value into the field's configured range."""
        if self.minimum is not None:
            number = max(number, self.minimum)
        if self.maximum is not None:
            number = min(number, self.maximum)
        return number

    def try_coerce(self, value):
        """Coerce without raising. Returns (ok, coerced_value)."""
        try:
            return True, self.coerce(value)
        except (ValueError, TypeError, OverflowError):
            return False, None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """Convert the field definition to a plain dictionary."""
        result = {
            'name': self.name,
            'type': self.type,
            'label': self.label,
            'default': self.default,
            'description': self.description,
            'visible': self.visible,
        }
        # Only emit the type-specific keys that actually apply, so the exported
        # YAML stays readable instead of carrying a wall of nulls.
        if self.type in ('choice', 'multichoice'):
            result['options'] = list(self.options)
        if self.type in ('int', 'float'):
            result['minimum'] = self.minimum
            result['maximum'] = self.maximum
            result['step'] = self.step
        if self.type == 'float':
            result['decimals'] = self.decimals
        if self.type in ('string', 'text') and self.max_length:
            result['max_length'] = self.max_length
        if self.taglab_key:
            result['taglab_key'] = self.taglab_key
        return result

    @classmethod
    def from_dict(cls, data):
        """Create a field definition from a plain dictionary."""
        return cls(
            name=data.get('name'),
            type=data.get('type', 'string'),
            label=data.get('label'),
            default=data.get('default'),
            options=data.get('options'),
            minimum=data.get('minimum'),
            maximum=data.get('maximum'),
            step=data.get('step'),
            decimals=data.get('decimals', 2),
            max_length=data.get('max_length', 0),
            description=data.get('description', ''),
            visible=data.get('visible', True),
            taglab_key=data.get('taglab_key'),
        )

    def __repr__(self):
        """Return a debug representation of the field."""
        return f"MetaDataField(name={self.name!r}, type={self.type!r}, default={self.default!r})"


class MetaDataSchema:
    """The project-level set of metadata field definitions.

    Stored once per project rather than once per annotation. Values live on
    annotations in their ``metadata`` dict, and only when they differ from the
    field default -- every read and write goes through get_value/set_value so
    that sparseness is impossible to bypass by accident.
    """

    def __init__(self, fields=None):
        """Initialize the schema, optionally with a list of MetaDataField objects."""
        self.fields = list(fields) if fields else []

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self):
        """Return the number of fields in the schema."""
        return len(self.fields)

    def __iter__(self):
        """Iterate over the fields in display order."""
        return iter(self.fields)

    def get_field(self, name):
        """Return the field with the given name, or None."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def has_field(self, name):
        """Return True if a field with the given name exists."""
        return self.get_field(name) is not None

    def visible_fields(self):
        """Return the fields flagged visible, in display order."""
        return [field for field in self.fields if field.visible]

    def taglab_fields(self):
        """Return the fields that map onto a native TagLab slot."""
        return [field for field in self.fields if field.taglab_key]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_field(self, field):
        """Append a field, rejecting a duplicate name."""
        if self.has_field(field.name):
            raise ValueError(f"A metadata field named '{field.name}' already exists.")
        self.fields.append(field)
        return field

    def remove_field(self, name):
        """Remove a field from the schema. Returns the removed field, or None."""
        field = self.get_field(name)
        if field is not None:
            self.fields.remove(field)
        return field

    def move_field(self, name, offset):
        """Move a field up or down in display order. Returns True if it moved."""
        field = self.get_field(name)
        if field is None:
            return False
        index = self.fields.index(field)
        new_index = index + offset
        if not 0 <= new_index < len(self.fields):
            return False
        self.fields.pop(index)
        self.fields.insert(new_index, field)
        return True

    # ------------------------------------------------------------------
    # Value access -- the sparse-store chokepoint
    # ------------------------------------------------------------------

    def get_value(self, annotation, name):
        """Read a field's value for an annotation, falling back to the default."""
        field = self.get_field(name)
        if field is None:
            return None
        stored = getattr(annotation, 'metadata', None)
        if not stored or name not in stored:
            return copy.deepcopy(field.default)
        return stored[name]

    def set_value(self, annotation, name, value):
        """Write a field's value, storing nothing when it equals the default.

        Returns True if the annotation's stored metadata actually changed.
        """
        field = self.get_field(name)
        if field is None:
            return False

        coerced = field.coerce(value)

        # Annotations created before this attribute existed (or restored from an
        # old pickle) may not carry the dict yet.
        if getattr(annotation, 'metadata', None) is None:
            annotation.metadata = {}

        if coerced == field.default:
            # Storing a default would bloat the project for no information gain.
            return annotation.metadata.pop(name, _MISSING) is not _MISSING

        if annotation.metadata.get(name, _MISSING) == coerced:
            return False

        annotation.metadata[name] = coerced
        return True

    def has_stored_value(self, annotation, name):
        """Return True if the annotation explicitly stores this field."""
        return name in (getattr(annotation, 'metadata', None) or {})

    def count_stored(self, annotations, name):
        """Count annotations explicitly storing a value for this field."""
        return sum(1 for annotation in annotations if self.has_stored_value(annotation, name))

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def prune(self, annotations, name):
        """Remove a field's stored values from every annotation. Returns the count."""
        removed = 0
        for annotation in annotations:
            stored = getattr(annotation, 'metadata', None)
            if stored and name in stored:
                del stored[name]
                removed += 1
        return removed

    def rename(self, annotations, old_name, new_name):
        """Migrate stored values from one field name to another. Returns the count."""
        if old_name == new_name:
            return 0
        migrated = 0
        for annotation in annotations:
            stored = getattr(annotation, 'metadata', None)
            if stored and old_name in stored:
                stored[new_name] = stored.pop(old_name)
                migrated += 1
        return migrated

    def recoerce(self, annotations, field):
        """Re-coerce stored values after a field's type or range changed.

        Values that cannot be represented under the new definition are dropped,
        as are values that now equal the default. Returns (kept, discarded).
        """
        kept = 0
        discarded = 0
        for annotation in annotations:
            stored = getattr(annotation, 'metadata', None)
            if not stored or field.name not in stored:
                continue
            ok, value = field.try_coerce(stored[field.name])
            if not ok or value == field.default:
                del stored[field.name]
                discarded += 1
            else:
                stored[field.name] = value
                kept += 1
        return kept, discarded

    def count_uncoercible(self, annotations, field):
        """Count stored values that the given field definition would discard."""
        count = 0
        for annotation in annotations:
            stored = getattr(annotation, 'metadata', None)
            if not stored or field.name not in stored:
                continue
            ok, _ = field.try_coerce(stored[field.name])
            if not ok:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Promotion from annotation.data
    # ------------------------------------------------------------------

    def promote_from_data(self, annotations):
        """Turn scalar keys in each annotation's ``data`` into schema fields.

        Importers dump arbitrary keys into ``annotation.data``, where they are
        invisible and uneditable. This moves every scalar one into a real typed
        field: the value is transferred to ``metadata`` and the key deleted from
        ``data``, so nothing is stored twice. Nested values (dicts, lists) are
        left behind for the read-only Raw Data group.

        Returns (added_field_names, promoted_value_count, dropped_key_count).
        """
        annotations = list(annotations)

        # Pass 1: decide on a field per key, using every annotation's value so a
        # column that is int in one row and str in another lands on str.
        observed = {}
        for annotation in annotations:
            for key, value in (getattr(annotation, 'data', None) or {}).items():
                if key in DERIVED_IMPORT_KEYS or not _is_scalar(value):
                    continue
                observed.setdefault(key, []).append(value)

        added = []
        for key, values in observed.items():
            if self.has_field(key):
                continue
            if key in SEEDED_FIELDS:
                field = MetaDataField(name=key, **SEEDED_FIELDS[key])
            else:
                field = MetaDataField(name=key, type=_infer_type(values), label=key)
            self.add_field(field)
            added.append(key)

        # Pass 2: move the values across.
        promoted = 0
        dropped = 0
        for annotation in annotations:
            data = getattr(annotation, 'data', None)
            if not data:
                continue
            for key in list(data.keys()):
                value = data[key]
                if key in DERIVED_IMPORT_KEYS:
                    # Recomputed on demand by the built-in tier; a stored copy
                    # would go stale the moment the geometry is edited.
                    del data[key]
                    dropped += 1
                    continue
                if not _is_scalar(value):
                    continue
                field = self.get_field(key)
                if field is None:
                    continue
                ok, coerced = field.try_coerce(value)
                if ok and self.set_value(annotation, key, coerced):
                    promoted += 1
                del data[key]

        return added, promoted, dropped

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """Convert the schema to a plain dictionary."""
        return {'fields': [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, data):
        """Create a schema from a plain dictionary. Tolerates None and bad rows."""
        if not data:
            return cls()
        # Accept either {'fields': [...]} or a bare list of field dicts.
        rows = data.get('fields', []) if isinstance(data, dict) else data
        fields = []
        for row in rows or []:
            try:
                fields.append(MetaDataField.from_dict(row))
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Skipping invalid metadata field definition {row!r}: {e}")
        return cls(fields)

    def to_yaml(self, file_path):
        """Write the schema to a YAML file."""
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(self.to_dict(), file, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path):
        """Read a schema from a YAML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return cls.from_dict(yaml.safe_load(file))

    def merge(self, other, replace_existing=False):
        """Merge another schema into this one by field name.

        Returns (added_names, skipped_names, replaced_names).
        """
        added, skipped, replaced = [], [], []
        for field in other:
            existing = self.get_field(field.name)
            if existing is None:
                self.fields.append(field)
                added.append(field.name)
            elif replace_existing:
                self.fields[self.fields.index(existing)] = field
                replaced.append(field.name)
            else:
                skipped.append(field.name)
        return added, skipped, replaced

    def __repr__(self):
        """Return a debug representation of the schema."""
        return f"MetaDataSchema(fields={[field.name for field in self.fields]})"


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def _is_scalar(value):
    """Return True if a value can be represented by a metadata field."""
    return value is None or isinstance(value, (str, int, float, bool))


def _infer_type(values):
    """Infer a field type from the values observed for an imported key."""
    seen = set()
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, bool):
            seen.add('bool')
        elif isinstance(value, int):
            seen.add('int')
        elif isinstance(value, float):
            seen.add('float')
        else:
            seen.add('string')

    if not seen:
        return 'string'
    if len(seen) == 1:
        return seen.pop()
    # Mixed numerics widen to float; anything else falls back to text.
    if seen <= {'int', 'float'}:
        return 'float'
    return 'string'
