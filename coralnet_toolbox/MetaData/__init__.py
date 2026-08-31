# coralnet_toolbox/MetaData/__init__.py

from .QtMetaDataSchema import MetaDataField
from .QtMetaDataSchema import MetaDataSchema
from .QtMetaDataSchema import FIELD_TYPES
from .QtMetaDataSchema import TYPE_DEFAULTS

__all__ = ["MetaDataField",
           "MetaDataSchema",
           "FIELD_TYPES",
           "TYPE_DEFAULTS"]

# NOTE: the dialog and built-in-field modules are deliberately NOT imported
# here. They pull in PyQt widgets, and this package is imported by the
# annotation IO layer, which must stay importable without a QApplication.
