# coralnet_toolbox/Tile/__init__.py

from .TileDataset.QtClassify import Classify as TileClassifyDataset
from .TileDataset.QtDetect import Detect as TileDetectDataset
from .TileDataset.QtSegment import Segment as TileSegmentDataset

__all__ = [
    'TileClassifyDataset',
    'TileDetectDataset',
    'TileSegmentDataset',
]