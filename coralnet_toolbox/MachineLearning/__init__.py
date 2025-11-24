# coralnet_toolbox/MachineLearning/__init__.py

from .Community.cfg import get_available_configs

from .TuneModel.QtClassify import Classify as TuneClassify
from .TuneModel.QtDetect import Detect as TuneDetect
from .TuneModel.QtSegment import Segment as TuneSegment
from .TuneModel.QtSemantic import Semantic as TuneSemantic

from .TrainModel.QtClassify import Classify as TrainClassify
from .TrainModel.QtDetect import Detect as TrainDetect 
from .TrainModel.QtSegment import Segment as TrainSegment
from .TrainModel.QtSemantic import Semantic as TrainSemantic

from .DeployModel.QtClassify import Classify as DeployClassify
from .DeployModel.QtDetect import Detect as DeployDetect
from .DeployModel.QtSegment import Segment as DeploySegment
from .DeployModel.QtSemantic import Semantic as DeploySemantic

from .VideoInference.QtDetect import Detect as VideoDetect
from .VideoInference.QtSegment import Segment as VideoSegment

from .ImportDataset.QtDetect import Detect as ImportDetect
from .ImportDataset.QtSegment import Segment as ImportSegment

from .ExportDataset.QtClassify import Classify as ExportClassify
from .ExportDataset.QtDetect import Detect as ExportDetect
from .ExportDataset.QtSegment import Segment as ExportSegment
from .ExportDataset.QtSemantic import Semantic as ExportSemantic

from .EvaluateModel.QtClassify import Classify as EvalClassify
from .EvaluateModel.QtDetect import Detect as EvalDetect
from .EvaluateModel.QtSegment import Segment as EvalSegment
from .EvaluateModel.QtSemantic import Semantic as EvalSemantic

from .MergeDatasets.QtClassify import Classify as MergeClassify
from .OptimizeModel.QtBase import Base as Optimize


__all__ = [
    'get_available_configs',
    "TuneClassify",
    "TuneDetect",
    "TuneSegment",
    "TuneSemantic",
    'TrainClassify', 
    'TrainDetect', 
    'TrainSegment',
    'TrainSemantic',
    'DeployClassify', 
    'DeployDetect', 
    'DeploySegment', 
    'DeploySemantic',
    'VideoDetect',
    'VideoSegment',
    'ImportDetect', 
    'ImportSegment',
    'ExportClassify', 
    'ExportDetect', 
    'ExportSegment',
    'ExportSemantic',
    'EvalClassify', 
    'EvalDetect', 
    'EvalSegment',
    'EvalSemantic',
    'TileDetect',
    'TileSegment',
    'MergeClassify', 
    'Optimize'
]