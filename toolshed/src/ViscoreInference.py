import os
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

from src.Common import get_now
from src.Common import IMG_FORMATS
from src.Common import console_user
from src.Common import progress_printer

from src.Patches import crop_patch

from src.Classification import get_validation_augmentation

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def get_shortcode_id_dict(file_path):
    """

    :param file_path:
    :return:
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    shortcode_id_dict = {}
    for item in data['classlist']:
        id_number, short_code, full_name = item
        shortcode_id_dict[short_code] = id_number

    return shortcode_id_dict


def viscore_inference(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("Viscore Inference")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get User classes dict
    if os.path.exists(args.user_json):
        with open(args.user_json, 'r') as file:
            user_classes_dict = json.load(file)
    else:
        raise Exception("ERROR: User JSON file not found.")

    # Get the QClasses dict
    if os.path.exists(args.qclasses_json):
        qclasses_dict = get_shortcode_id_dict(args.qclasses_json)
    else:
        raise Exception(f"ERROR: QClasses JSON File doesn't exist.")

    # Points Dataframe
    if os.path.exists(args.points):
        # Annotation file
        annotation_file = args.points
        points = pd.read_csv(annotation_file)
        # Check that the needed columns are in the dataframe
        assert "Row" in points.columns, print(f"ERROR: 'Row' not in provided csv")
        assert "Column" in points.columns, print(f"ERROR: 'Column' not in provided csv")
        assert "Name" in points.columns, print(f"ERROR: 'Name' not in provided csv")
        assert "Dot" in points.columns, print(f"ERROR: 'Dot' not in provided csv")
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
        raise Exception("ERROR: Points provided doesn't exist.")

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
        raise Exception("ERROR: Directory provided doesn't exist.")

    # Model Weights
    if os.path.exists(args.model):
        try:
            # Load into the model
            model = torch.load(args.model)
            model_name = model.name
            print(f"NOTE: Loaded weights {model.name}")

            # Get the preprocessing function that was used during training
            preprocessing_fn = smp.encoders.get_preprocessing_fn(model_name, 'imagenet')

            # Convert patches to PyTorch tensor with validation augmentation and preprocessing
            validation_augmentation = get_validation_augmentation(height=224, width=224)

            # Set the model to evaluation mode
            model.eval()

        except Exception as e:
            raise Exception(f"ERROR: There was an issue loading the model\n{e}")

    else:
        raise Exception("ERROR: Model provided doesn't exist.")

    # Class map
    if os.path.exists(args.class_map):
        with open(args.class_map, 'r') as json_file:
            class_mapping_dict = json.load(json_file)

        class_map = list(class_mapping_dict.keys())
        num_classes = len(class_map)

    else:
        raise Exception(f"ERROR: Class Map file provided doesn't exist.")

    # Output
    output_dir = f"{args.output_dir}/predictions/"
    output_csv = f"{output_dir}classifier_{get_now()}_predictions.csv"
    output_json = f"{output_dir}classifier_{get_now()}_samples.cl.user.robot.json"
    os.makedirs(output_dir, exist_ok=True)

    # Output points
    output_points = pd.DataFrame(columns=points.columns.tolist())

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
        print(f"\nNOTE: Cropping patches ({patch_size} x {patch_size}) for {image_name}")

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
        patches = [torch.Tensor(preprocessing_fn(validation_augmentation(image=patch)['image'])) for patch in patches]
        patches_tensor = torch.stack(patches).permute(0, 3, 1, 2)
        patches_tensor = patches_tensor.to(device)

        # ----------------------------------------------------------------
        # Inference
        # ----------------------------------------------------------------

        # Model to make predictions
        print(f"NOTE: Making predictions on patches for {image_name}")
        with torch.no_grad():
            probabilities = model(patches_tensor)

        # Convert PyTorch tensor to numpy array
        probabilities = probabilities.cpu().numpy()
        # Get predicted class indices
        predictions = np.argmax(probabilities, axis=1)
        class_predictions = np.array([class_map[v] for v in predictions]).astype(str)

        # Make a copy
        image_predictions = image_points.copy()

        N = probabilities.shape[1]
        sorted_prob_indices = np.argsort(probabilities, axis=1)[:, ::-1]
        top_N_confidences = probabilities[np.arange(probabilities.shape[0])[:, np.newaxis], sorted_prob_indices[:, :N]]
        top_N_suggestions = np.array([[class_map[i] for i in indices] for indices in sorted_prob_indices[:, :N]])

        # CoralNet format only goes to the first 5 choices
        if num_classes > 5:
            num_classes = 5

        for index, class_num in enumerate(range(num_classes)):
            image_predictions['Machine confidence ' + str(index + 1)] = top_N_confidences[:, index]
            image_predictions['Machine suggestion ' + str(index + 1)] = top_N_suggestions[:, class_num]

        output_points = pd.concat((output_points, image_predictions))

    # Update the classes in user dict, using Dot ID
    for index in range(len(user_classes_dict['cl'])):
        try:
            dot_df = output_points[output_points['Dot'] == index]

            # For debugging, shouldn't happen
            if dot_df.empty:
                continue

            # Get the top one for all views, get the mode; get the confidence
            mode_class = dot_df['Machine suggestion 1'].mode().item()
            mode_conf = dot_df[dot_df['Machine suggestion 1'] == mode_class]['Machine confidence 1'].mean()

            if not mode_class in qclasses_dict:
                raise Exception(f"Mode class {mode_class} not found in QClasses")

            if int(mode_conf * 100) < args.conf:
                raise Exception(f"Confidence for Dot {index} below {args.conf}")

            mode_id = qclasses_dict[mode_class]

        except Exception as e:
            print(f"WARNING: {e}; setting as 'Review'")
            mode_id = qclasses_dict['Review']

        user_classes_dict['cl'][index] = mode_id

    # Save image predictions to CSV
    output_points.sort_index(inplace=True)
    output_points.reset_index(drop=True, inplace=True)
    output_points.to_csv(output_csv)

    # Save image predictions to JSON
    with open(output_json, "w") as json_file:
        json.dump(user_classes_dict, json_file, indent=4)

    print(f"NOTE: Predictions saved to {output_dir}")


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

    parser.add_argument('--user_json', type=str, required=True,
                        help='An empty User JSON file for the plot')

    parser.add_argument('--qclasses_json', type=str, required=True,
                        help='A QClasses JSON file for the plot')

    parser.add_argument("--model", type=str, required=True,
                        help="Path to Best Model and Weights File (.pth)")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the model's Class Map JSON file")

    parser.add_argument("--conf", type=int, default=50,
                        help="Confidence threshold value")

    parser.add_argument("--patch_size", type=int, default=112,
                        help="The size of each patch extracted")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:

        viscore_inference(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()