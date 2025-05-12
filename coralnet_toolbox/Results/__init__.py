"""
CoralNet Toolbox Results Module

This module contains classes for handling prediction results from various models,
converting between formats, and efficiently storing and processing result data.
"""

from coralnet_toolbox.Results.CombineResults import CombineResults
from coralnet_toolbox.Results.ConvertResults import ConvertResults
from coralnet_toolbox.Results.MapResults import MapResults
from coralnet_toolbox.Results.Masks import Masks
from coralnet_toolbox.Results.ResultsProcessor import ResultsProcessor

__all__ = [
    'CombineResults',
    'ConvertResults',
    'MapResults', 
    'Masks',
    'ResultsProcessor'
]