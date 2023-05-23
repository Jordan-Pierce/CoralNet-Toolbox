import os
import glob
from tqdm import tqdm

import math
import numpy as np
import pandas as pd
from skimage.io import imread, imsave

from keras import backend as K

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------
IMAGE_FORMATS = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------

def crop(image, y, x, size=112):
    """
    Given an image, and a Y, X location, this function will extract the patch.
    Default patch size is 112 x 112
    """

    patch = image[abs(size - y): abs(size + y), abs(size - x): abs(size + x), :]

    return patch


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
