# coralnet_toolbox/MachineLearning/__init__.py

from .Community.cfg import get_available_configs

from .TrainModel.QtClassify import Classify as TrainClassify
from .TrainModel.QtDetect import Detect as TrainDetect 
from .TrainModel.QtSegment import Segment as TrainSegment

from .DeployModel.QtClassify import Classify as DeployClassify
from .DeployModel.QtDetect import Detect as DeployDetect
from .DeployModel.QtSegment import Segment as DeploySegment

from .BatchInference.QtClassify import Classify as BatchClassify
from .BatchInference.QtDetect import Detect as BatchDetect
from .BatchInference.QtSegment import Segment as BatchSegment

from .ImportDataset.QtDetect import Detect as ImportDetect
from .ImportDataset.QtSegment import Segment as ImportSegment

from .ExportDataset.QtClassify import Classify as ExportClassify
from .ExportDataset.QtDetect import Detect as ExportDetect
from .ExportDataset.QtSegment import Segment as ExportSegment

from .EvaluateModel.QtClassify import Classify as EvalClassify
from .EvaluateModel.QtDetect import Detect as EvalDetect
from .EvaluateModel.QtSegment import Segment as EvalSegment

from .MergeDatasets.QtClassify import Classify as MergeClassify
from .OptimizeModel.QtBase import Base as Optimize

__all__ = [
    'get_available_configs',
    'TrainClassify', 
    'TrainDetect', 
    'TrainSegment',
    'DeployClassify', 
    'DeployDetect', 
    'DeploySegment', 
    'BatchClassify',
    'BatchDetect', 
    'BatchSegment',
    'ImportDetect', 
    'ImportSegment',
    'ExportClassify', 
    'ExportDetect', 
    'ExportSegment',
    'EvalClassify', 
    'EvalDetect', 
    'EvalSegment',
    'TileDetect',
    'TileSegment',
    'MergeClassify', 
    'Optimize'
]