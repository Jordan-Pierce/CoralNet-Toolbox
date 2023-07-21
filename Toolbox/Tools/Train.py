import os
import glob

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import *

from plot_keras_history import plot_history
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from faker import Faker
from imgaug import augmenters as iaa

from . import *


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def compute_class_weights(df, mu=0.15):
    """
    Compute class weights for the given dataframe.
    """
    # Compute the value counts for each class
    value_counts = df['Label'].value_counts().to_dict()
    total = sum(value_counts.values())
    keys = value_counts.keys()

    # To store the class weights
    class_weight = dict()

    # Compute the class weights for each class
    for key in keys:
        score = math.log(mu * total / float(value_counts[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


def recall_m(y_true, y_pred):
    """
    Compute the recall metric.
    """
    # Compute the true positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # Compute the possible positives
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # Compute the recall
    recall = true_positives / (possible_positives + K.epsilon())

    return recall


def precision_m(y_true, y_pred):
    """
    Compute the precision metric.
    """
    # Compute the true positives
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # Compute the predicted positives
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # Compute the precision
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision