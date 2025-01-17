# coralnet_toolbox/Tile/__init__.py

from .TileDataset.QtDetect import Detect as TileDetectDataset
from .TileDataset.QtSegment import Segment as TileSegmentDataset
from .TileInference.QtDetect import Detect as TileDetectInference
from .TileInference.QtSegment import Segment as TileSegmentInference

__all__ = [
    'TileDetectDataset',
    'TileSegmentDataset',
    'TileDetectInference',
    'TileSegmentInference',
]
