import os
import glob
from tqdm import tqdm
import concurrent.futures

import math
import numpy as np
import pandas as pd
from skimage.io import imread, imsave

from tensorflow.keras import backend as K


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------
IMAGE_FORMATS = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]

# -------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------


def crop_patch(image, y, x, patch_size=224):
    """
    Given an image, and a Y, X location, this function will extract the patch.
    """

    height, width, _ = image.shape

    # N x N
    size = int(patch_size // 2)

    # If padding is needed
    top_pad = 0
    bottom_pad = 0
    right_pad = 0
    left_pad = 0

    top = y - size
    if top < 0:
        top_pad = abs(top)
        top = 0

    bottom = y + size
    if bottom > height:
        bottom_pad = bottom - height
        bottom = height

    left = x - size
    if left < 0:
        left_pad = abs(left)
        left = 0

    right = x + size
    if size > width:
        right_pad = right - width
        right = width

    # Get the sub-image from image
    patch = image[top: bottom, left: right, :]

    # Check if the sub-image size is smaller than N x N
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:

        # Pad the sub-image with zeros
        patch = np.pad(patch, ((top_pad, bottom_pad),
                               (left_pad, right_pad),
                               (0, 0)))

    # Check if the percentage of padded zeros is more than 50%
    if np.mean(patch == 0) > 0.75:
        patch = None

    return patch


def crop_patches(image_df, output_dir):
    """
    Given an image dataframe, this function will crop a patch for each annotation
    """

    # Setup
    image_name = image_df['Name'].iloc[0]
    image_prefix = image_name.split(".")[0]
    image_path = image_df['Image Path'].iloc[0]
    image = imread(image_path)
    processed_patches = []

    # Loop through each annotation for this image
    for i, r in image_df.iterrows():

        try:
            # Extract the patch
            patch = crop_patch(image, r['Row'], r['Column'])
            name = f"{image_prefix}_{r['Row']}_{r['Column']}_{r['Label']}"
            path = output_dir + r['Label'] + "/" + name + ".png"
            # If it's not mostly empty, crop it
            if not os.path.exists(path) and patch is not None:
                imsave(fname=path, arr=patch)
                processed_patches.append([name, path, r['Label'], image_name, image_path])
            else:
                processed_patches.append([None, None, r['Label'], image_name, image_path])

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return processed_patches


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
