import os
import sys
import json
import glob
import argparse
import warnings
import traceback

import numpy as np
import pandas as pd
from skimage.io import imread

import torch
import torchvision

import segmentation_models_pytorch as smp

from Common import get_now
from Common import IMG_FORMATS
from Common import progress_printer

from Patches import crop_patch

from Classification import get_preprocessing
from Classification import get_validation_augmentation


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def get_class_map(path):
    """

    """
    with open(path, 'r') as json_file:
        class_mapping_dict = json.load(json_file)

    return class_mapping_dict


def image_inference(args):
    """

    """
    print("\n###############################################")
    print("Classification Inference")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Points Dataframe
    if os.path.exists(args.points):
        # Annotation file
        annotation_file = args.points
        points = pd.read_csv(annotation_file)
        # Check that the needed columns are in the dataframe
        assert "Row" in points.columns, print(f"ERROR: 'Row' not in provided csv")
        assert "Column" in points.columns, print(f"ERROR: 'Column' not in provided csv")
        assert "Name" in points.columns, print(f"ERROR: 'Name' not in provided csv")
        # Redundant, just in case user passes the file path instead of the file name
        points['Name'] = [os.path.basename(n) for n in points['Name'].values]
        # Get the names of all the images
        image_names = np.unique(points['Name'].to_numpy())
        image_names = [os.path.basename(n) for n in image_names]
        if image_names:
            print(f"NOTE: Found a total of {len(points)} sampled points for {len(points['Name'].unique())} images")
        else:
            raise Exception(f"ERROR: Issue with 'Name' column; check input provided")
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

            # Load into the model
            model = torch.load(args.model)
            model_name = "-".join(model.name.split("-")[1:])
            print(f"NOTE: Loaded weights {model.name}")

            # Get the preprocessing function that was used during training
            preprocessing_fn = smp.encoders.get_preprocessing_fn(model_name, 'imagenet')

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
    output_dir = f"{args.output_dir}\\predictions\\"
    output_path = f"{output_dir}classifier_{get_now()}_predictions.csv"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Inference Loop
    # ----------------------------------------------------------------

    # Loop through each of the images
    for n_idx, image_file in progress_printer(enumerate(image_files)):

        # Get the image name
        image_name = os.path.basename(image_file)
        # Open the image
        image = imread(image_file)

        # ----------------------------------------------------------------
        # Creating patches
        # ----------------------------------------------------------------
        patch_size = args.patch_size

        # To hold all the patches (memory)
        patches = []

        # Create patches for this image
        print(f"NOTE: Cropping patches ({patch_size} x {patch_size}) for {image_name}")

        # Get the current image points
        image_points = points[points['Name'] == image_name]

        # Make sure it's not empty for some reason
        if image_points.empty:
            print(f"ERROR: No image points found for {image_name}")
            continue

        for i, r in image_points.iterrows():
            patches.append(crop_patch(image, r['Row'], r['Column'], patch_size))

        # Convert to numpy array
        patches = np.stack(patches)

        # Convert patches to PyTorch tensor with validation augmentation and preprocessing
        validation_augmentation = get_validation_augmentation(height=224, width=224)
        preprocessing = get_preprocessing(preprocessing_fn=preprocessing_fn)

        patches_tensor = torch.stack([torch.Tensor(preprocessing(validation_augmentation(patch))) for patch in patches])
        patches_tensor = patches_tensor.to(device)

        # Set the model to evaluation mode
        model.eval()

        # ----------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------

        # Model to make predictions
        print(f"NOTE: Making predictions on patches for {image_name}")
        with torch.no_grad():
            probabilities = model(patches_tensor)

        # Get predicted class indices
        _, predictions = torch.max(probabilities, 1)

        # Convert PyTorch tensor to numpy array
        predictions = predictions.cpu().numpy()
        class_predictions = np.array([class_map[str(v)] for v in predictions]).astype(str)

        # Make a copy
        output = image_points.copy()

        N = probabilities.shape[1]
        sorted_prob_indices = np.argsort(probabilities, axis=1)[:, ::-1]
        top_N_confidences = probabilities[np.arange(probabilities.shape[0])[:, np.newaxis], sorted_prob_indices[:, :N]]
        top_N_suggestions = np.array([[class_map[str(i)] for i in indices] for indices in sorted_prob_indices[:, :N]])

        # CoralNet format only goes to the first 5 choices
        num_classes = model.layers[-1].output_shape[-1]

        if num_classes > 5:
            num_classes = 5

        for index, class_num in enumerate(range(num_classes)):
            output['Machine confidence ' + str(index + 1)] = top_N_confidences[:, index]
            output['Machine suggestion ' + str(index + 1)] = top_N_suggestions[:, class_num]

        # Save with previous prediction, if any exists
        if os.path.exists(output_path):
            previous_predictions = pd.read_csv(output_path, index_col=0)
            output = pd.concat((previous_predictions, output))

        # Save each image predictions
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
                        help="Path to Best Model and Weights File (.pth)")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the model's Class Map JSON file")

    parser.add_argument("--patch_size", type=int, default=112,
                        help="The size of each patch extracted")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        image_inference(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()