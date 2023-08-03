import os
import sys
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

from Toolbox.Tools import *
from Toolbox.Tools.Patches import crop_patch
from Toolbox.Tools.Classifier import precision, recall, f1_score


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def get_class_map(path):
    """

    """
    with open(path, 'r') as json_file:
        class_mapping_dict = json.load(json_file)

    # Reverse the keys and values to create a mapping of class names to indices
    class_map = {v: k for k, v in class_mapping_dict.items()}

    return class_map


def inference(args):
    """

    """
    print("\n###############################################")
    print("Inference")
    print("###############################################\n")

    # Set the variables

    # Image files
    if os.path.exists(args.images):
        images = [i for i in glob.glob(f"{args.images}/*.*") if i.split(".")[-1].lower() in IMG_FORMATS]
        if not images:
            raise Exception(f"ERROR: No images were found in the directory provided; please check input.")
        else:
            print(f"NOTE: Found {len(images)} images in directory provided")
    else:
        print("ERROR: Directory provided doesn't exist.")
        sys.exit(1)

    # Points Dataframe
    if os.path.exists(args.points):
        points = pd.read_csv(args.points)
        print(f"NOTE: Found a total of {len(points)} sampled points for {len(points['Name'].unique())} images")
    else:
        print("ERROR: Points provided doesn't exist.")
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
    output_path = f"{output_dir}\predictions.csv"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    print("NOTE: Making predictions...")

    output = []
    patches = []

    # Subset the images list to only contain those with points
    images = [i for i in images if os.path.basename(i) in points['Name'].unique()]

    # Loop through each image, extract the corresponding patches
    for idx, image_path in enumerate(images):

        # Get the points associated
        name = os.path.basename(image_path)
        current_points = points[points['Name'] == name]

        if current_points.empty:
            continue

        # Read the image
        image = imread(image_path)

        # Crop patches from points
        for i, r in current_points.iterrows():
            patches.append(crop_patch(image, r['Row'], r['Column'], patch_size=224))
            output.append(r)

        # Gooey
        print_progress(idx, len(images))

    # Convert to numpy array
    patches = np.array(patches)

    # Make predictions for this image
    probabilities = model.predict(patches)
    predictions = np.argmax(probabilities, axis=1)
    class_predictions = np.array([class_map[v] for v in predictions])

    # Convert, make the top choice Label
    output = pd.DataFrame(output)
    output['Label'] = class_predictions

    N = probabilities.shape[1]  # Number of classes
    sorted_prob_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    top_N_confidences = probabilities[np.arange(probabilities.shape[0])[:, np.newaxis], sorted_prob_indices[:, :N]]
    top_N_suggestions = np.array([[class_map[idx] for idx in indices] for indices in sorted_prob_indices[:, :N]])

    # CoralNet format only goes to the first 5 choices
    for index, class_num in enumerate(range(5)):
        rounded_confidences = np.around(top_N_confidences[:, index], decimals=4)
        output['Machine confidence ' + str(index + 1)] = rounded_confidences
        output['Machine suggestion ' + str(index + 1)] = top_N_suggestions[:, class_num]

    # Save with previous prediction, if any exists
    if os.path.exists(output_path):
        previous_predictions = pd.read_csv(output_path, index_col=0)
        output = pd.concat((previous_predictions, output))

    # Save
    output.to_csv(output_path)
    print(f"NOTE: Predictions saved to {output_path}")


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
