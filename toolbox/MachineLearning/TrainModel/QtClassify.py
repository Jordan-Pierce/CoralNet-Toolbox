import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import datetime
import gc
import json
import os
from pathlib import Path

import ultralytics.data.build as build
import ultralytics.models.yolo.classify.train as train_build

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QScrollArea, QMessageBox, QCheckBox, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QDialog, QHBoxLayout, QPushButton, QComboBox, QSpinBox,
                             QFormLayout, QTabWidget, QDoubleSpinBox)

from torch.cuda import empty_cache
from ultralytics import YOLO

from toolbox.MachineLearning.WeightedDataset import WeightedInstanceDataset
from toolbox.MachineLearning.WeightedDataset import WeightedClassificationDataset
from toolbox.MachineLearning.QtEvaluateModel import EvaluateModelWorker

from toolbox.MachineLearning.TrainModel.QtBase import Base
from toolbox.MachineLearning.TrainModel.QtBase import TrainModelWorker


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Classify(Base):
    pass