import os
import gc
import sys
import shutil
import warnings
import argparse
import subprocess
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import glob
import math
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt

import tensorflow as tf

keras = tf.keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from Toolbox.Tools import *
from Toolbox.Tools.Patches import *
from Toolbox.Tools.Classifier import precision, recall, f1_score


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def get_class_map(path):
    """

    """
    with open(path, 'r') as json_file:
        class_mapping_dict = json.load(json_file)

    return class_mapping_dict


def inference(args):
    """

    """
    print("\n###############################################")
    print("Inference")
    print("###############################################\n")

    # Check that the user has GPU available
    if tf.config.list_physical_devices('GPU'):
        print("NOTE: Found GPU")
    else:
        print("WARNING: No GPU found; defaulting to CPU")

    # Points Dataframe
    if os.path.exists(args.points):
        # Annotation file
        annotation_file = args.points
        points = pd.read_csv(annotation_file, index_col=0)
        image_names = np.unique(points['Name'].to_numpy())
        print(f"NOTE: Found a total of {len(points)} sampled points for {len(points['Name'].unique())} images")
    else:
        print("ERROR: Points provided doesn't exist.")
        sys.exit(1)

    # Image files
    if os.path.exists(args.images):
        # Directory containing images
        image_dir = args.images
        # Get images from directory
        image_files = [i for i in glob.glob(f"{image_dir}/*.*") if i.split(".")[-1].lower() in IMG_FORMATS]
        # Subset the images list to only contain those with points
        image_files = [i for i in image_files if os.path.basename(i) in image_names]

        if not image_files:
            raise Exception(f"ERROR: No images were found in the directory provided; please check input.")
        else:
            print(f"NOTE: Found {len(image_files)} images in directory provided")
    else:
        print("ERROR: Directory provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    if os.path.exists(args.model):
        try:
            # Load the model with custom metrics
            custom_objects = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
            model = load_model(args.model, custom_objects=custom_objects)
            print(f"NOTE: Loaded model {args.model}")

        except Exception as e:
            print(f"ERROR: There was an issue loading the model\n{e}")
            sys.exit(1)
    else:
        print("ERROR: Model provided doesn't exist.")
        sys.exit(1)

    # Class map
    if os.path.exists(args.class_map):
        class_map = get_class_map(args.class_map)
    else:
        print(f"ERROR: Class Map file provided doesn't exist.")
        sys.exit()

    # Output
    output_dir = args.output_dir
    output_path = f"{output_dir}\\predictions.csv"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Inference Loop
    # ----------------------------------------------------------------

    # Loop through each of the images
    for n_idx, image_file in enumerate(image_files):

        # Get the image name
        image_name = os.path.basename(image_file)
        # Open the image
        image = imread(image_file)

        # ----------------------------------------------------------------
        # Creating patches
        # ----------------------------------------------------------------

        # To hold all the patches (memory)
        patches = []

        # Create patches for this image
        print(f"NOTE: Cropping patches for {image_name}")
        # Get the current image points
        image_points = points[points['Name'] == image_name]

        for i, r in image_points.iterrows():
            patches.append(crop_patch(image, r['Row'], r['Column'], 224))

        # Convert to numpy array
        patches = np.stack(patches)

        # ----------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------

        # Model to make predictions
        print(f"NOTE: Making predictions on patches for {image_name}")
        probabilities = model.predict(patches, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        class_predictions = np.array([class_map[str(v)] for v in predictions]).astype(str)

        # Convert, make the top choice Label
        output = image_points.copy()
        output['Label'] = class_predictions

        N = probabilities.shape[1]
        sorted_prob_indices = np.argsort(probabilities, axis=1)[:, ::-1]
        top_N_confidences = probabilities[np.arange(probabilities.shape[0])[:, np.newaxis], sorted_prob_indices[:, :N]]
        top_N_suggestions = np.array([[class_map[str(idx)] for idx in indices] for indices in sorted_prob_indices[:, :N]])

        # CoralNet format only goes to the first 5 choices
        for index, class_num in enumerate(range(5)):
            rounded_confidences = np.around(top_N_confidences[:, index], decimals=4)
            output['Machine confidence ' + str(index + 1)] = rounded_confidences
            output['Machine suggestion ' + str(index + 1)] = top_N_suggestions[:, class_num]

        # Save with previous prediction, if any exists
        if os.path.exists(output_path):
            previous_predictions = pd.read_csv(output_path, index_col=0)
            output = pd.concat((previous_predictions, output))

        # Save each image predictions
        output.to_csv(output_path)
        print(f"NOTE: Predictions saved to {output_path}")

        print_progress(n_idx, len(image_names))

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Model Inference")

    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--points", type=str, required=True,
                        help="Path to the points file containing 'Name', 'Row', and 'Column' information.")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to Best Model and Weights File (.h5)")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the model's Class Map JSON file")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        inference(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
