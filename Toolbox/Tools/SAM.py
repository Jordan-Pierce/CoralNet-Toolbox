import os
import sys
import requests
from tqdm import tqdm

import numpy as np
from skimage.io import imread
from skimage.io import imsave

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torchvision

from segment_anything import SamPredictor
from segment_anything import sam_model_registry

from Toolbox.Tools import *
from Toolbox.Tools.Inference import get_class_map

cmap = cm.get_cmap('tab20')

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def download_file(url, path):
    """

    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                # Write the content to the file
                file.write(response.content)
            print(f"NOTE: Downloaded file successfully")
            print(f"NOTE: Saved file to {path}")
        else:
            print(f"ERROR: Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred: {e}")


def get_sam_predictor(model_type="vit_b", device='cpu'):
    """

    """
    # URL to download pre-trained weights
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/"

    # The path containing the weights
    sam_root = os.path.abspath("./SAM_Weights/")
    os.makedirs(sam_root, exist_ok=True)

    # Mapping between the model type, and the checkpoint file name
    sam_dict = {"vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"}

    if model_type not in list(sam_dict.keys()):
        print(f"ERROR: Invalid model type provided; choices are:\n{list(sam_dict.keys())}")
        sys.exit(1)

    # Checkpoint path to model
    path = f"{sam_root}\\{sam_dict[model_type]}"

    # Check to see if the weights of the model type were already downloaded
    if not os.path.exists(path):
        print("NOTE: Model checkpoint does not exist; downloading")
        url = f"{sam_url}{sam_dict[model_type]}"

        # Download the file
        download_file(url, path)

    # Loading the mode, returning the predictor
    sam_model = sam_model_registry[model_type](checkpoint=path)
    sam_model.to(device=device)
    sam_predictor = SamPredictor(sam_model)

    return sam_predictor


def show_mask(mask, ax, random_color=False):
    """

    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def mode(a, background_class=255, axis=0):
    """
    Scipy's code to calculate the statistical mode of an array
    Here we include the ability to input a NULL value that should be ignored
    So in no situation should an index in the resulting dense annotations contain
    the background/NULL value.
    a -> the stack containing multiple 2-d arrays with the same dimensions
    """

    a = np.array(a)
    scores = np.unique(np.ravel(a))
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape, dtype=int)
    oldcounts = np.zeros(testshape, dtype=int)

    try:

        for score in scores:

            # if the mode is background_class,
            # use the second most common value instead
            if score == background_class:
                continue

            template = (a == score)
            counts = np.expand_dims(np.sum(template, axis), axis)

            mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = np.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        mostfrequent = mostfrequent[0]

    except Exception as e:
        print(f"ERROR: Could not calculate the mode accoss channels.\n{e}")
        mostfrequent = None

    return mostfrequent


def mss_sam(args):
    """

    """
    print("\n###############################################")
    print("Multilevel Superpixel Segmentation w/ SAM")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # Predictions Dataframe
    if os.path.exists(args.annotations):
        points = pd.read_csv(args.annotations)
        print(f"NOTE: Found a total of {len(points)} sampled points for {len(points['Name'].unique())} images")
    else:
        print("ERROR: Points file provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    try:
        # Load the model with custom metrics
        sam_predictor = get_sam_predictor(args.model_type, device)
        print(f"NOTE: Loaded model {args.model_type}")

    except Exception as e:
        print(f"ERROR: There was an issue loading the model\n{e}")
        sys.exit(1)

    # Class map
    if os.path.exists(args.class_map):
        class_map = get_class_map(args.class_map)
    else:
        print(f"ERROR: Class Map file provided doesn't exist.")
        sys.exit(1)

    # Setting output variables
    output_dir = args.output_dir
    mask_dir = f"{args.output_dir}masks/"
    mask_file = f"{output_dir}masks.csv"
    os.makedirs(mask_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    print("NOTE: Making predictions")

    output = []

    # Subset the images list to only contain those with points
    images = [i for i in images if os.path.basename(i) in points['Name'].unique()]

    # For plotting colored points and masks
    # Create a dictionary to map colors to labels using 'tab20' colormap
    unique_labels = points['Label'].unique()
    label_colors = {l: cmap(i) for i, l in enumerate(unique_labels)}

    # To hold all the cleans masks
    updated_masks = []

    # Loop through each image, extract the corresponding patches
    for i_idx, image_path in enumerate(images):

        # Get the points associated
        name = os.path.basename(image_path)
        current_points = points[points['Name'] == name]
        point_colors = current_points['Label'].map(label_colors).values

        # Skip if there are no points for some reason
        if current_points.empty:
            continue

        # Read the image, get the points, create SAM labels (background or foreground)
        image = imread(image_path)

        input_points = current_points[['Row', 'Column']].values.reshape(-1, 2)
        input_points = torch.Tensor(input_points).to(sam_predictor.device).unsqueeze(1)
        input_points = sam_predictor.transform.apply_coords_torch(input_points, image.shape[:2])

        input_labels = torch.Tensor([1 for _ in range(len(current_points))]).to(sam_predictor.device).unsqueeze(1)

        print(f"NOTE: Setting {name}")
        sam_predictor.set_image(image)

        for p_idx in tqdm(range(len(current_points))):

            masks, scores, logits = sam_predictor.predict_torch(point_coords=input_points[p_idx:p_idx+1],
                                                                point_labels=input_labels[p_idx:p_idx+1],
                                                                boxes=None,
                                                                multimask_output=True)
            # Numpy array masks
            masks = masks.cpu().detach().numpy().squeeze()
            scores = scores.cpu().detach().numpy().squeeze()

            try:
                for m_idx, mask in enumerate(masks):
                    if scores[m_idx] >= .75:
                        mask = mask.astype(np.uint8).squeeze()
                        mask[mask == 0] = 255
                        mask[mask == 1] = int(class_map[current_points['Label'].values[p_idx]])

                        updated_masks.append(mask)

            except Exception as e:
                print(f"WARNING: {e}")
                continue

        # Convert to numpy, find mode along channel axis
        updated_masks = np.array(updated_masks)
        mode_mask = mode(updated_masks)

        # Plot masks
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.scatter(current_points['Column'].values, current_points['Row'].values, c=point_colors)
        plt.imshow(mode_mask)
        plt.show()

        # Gooey
        print_progress(i_idx, len(images))


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Model Inference")

    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to the points file containing 'Name', 'Row', 'Column', and 'Label' information.")

    parser.add_argument("--model_type", type=str, default='vit_b',
                        help="Model to use; one of ['vit_b', 'vit_l', 'vit_h']")

    parser.add_argument("--class_map", type=str, required=True,
                        help="Path to the model's Class Map JSON file")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        mss_sam(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
