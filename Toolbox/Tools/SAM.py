import os
import sys
import requests
from tqdm import tqdm

import numpy as np
from skimage.io import imread
from skimage.io import imsave
from scipy.stats import mode as mode2d


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


def get_bbox(image, y, x, patch_size=224):
    """
    Given an image, and a Y, X location, this function will return a bounding box.
    """

    height, width, _ = image.shape

    # N x N
    size = patch_size // 2

    # Top of the patch, else edge of image
    top = y - size
    if top < 0:
        top = 0

    # Bottom of patch, else edge of image
    bottom = y + size
    if bottom > height:
        bottom = height

    # Left of patch, else edge of image
    left = x - size
    if left < 0:
        left = 0

    # Right of patch, else edge of image
    right = x + size
    if right > width:
        right = width

    # Bounding Box
    bbox = [left, top, right, bottom]

    return bbox


def find_most_common_label_in_area(points, binary_mask, bounding_box):
    """

    """
    # Get the coordinates of the bounding box
    min_x, min_y, max_x, max_y = bounding_box

    # Filter points within the bounding box
    points_in_area = points[(points['Column'] >= min_x) & (points['Column'] <= max_x) &
                            (points['Row'] >= min_y) & (points['Row'] <= max_y)]

    # Filter points that correspond to 1-valued regions in the binary mask
    mask_indices = points_in_area.apply(lambda row: binary_mask[row['Row'], row['Column']], axis=1)
    points_in_mask = points_in_area[mask_indices == 1]

    # Find the most common label
    most_common_label = mode2d(points_in_mask['Label'])[0][0]

    return most_common_label


def mode(a, background_class=255, axis=0):
    """
    Scipy's code to calculate the statistical mode of an array
    Here we include the ability to input a NULL value that should be ignored
    So in no situation should an index in the resulting dense annotations contain
    the background/NULL value.
    a -> the stack containing multiple 2-d arrays with the same dimensions
    """

    # If it's one channel
    if a.shape[2] == 1:
        return a

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


def colorize_mask(mask, class_map, label_colors):

    # Initialize the RGB mask with zeros
    height, width = mask.shape
    rgb_mask = np.full((height, width, 3), fill_value=255, dtype=np.uint8)

    cmap = {v: label_colors[k][0:3] for k, v in class_map.items()}

    for val in np.unique(mask):
        if val in class_map.values():
            color = np.array(cmap[val]) * 255
            rgb_mask[mask == val, :] = color.astype(np.uint8)

    return rgb_mask.astype(np.uint8)


def plot_mask(image, mask_color, points, point_colors, fname, mask_dir):
    """

    """

    # Plot masks
    plt.figure(figsize=(10, 10))
    plt.title(fname)
    plt.imshow(image)
    plt.imshow(mask_color, alpha=.75)
    plt.scatter(points['Column'].values, points['Row'].values, c=point_colors, s=1)
    plt.savefig(f"{mask_dir}{fname}")
    plt.close()


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
        points = pd.read_csv(args.annotations, index_col=0)
        print(f"NOTE: Found a total of {len(points)} sampled points for {len(points['Name'].unique())} images")
    else:
        print("ERROR: Points file provided doesn't exist.")
        sys.exit(1)

    # Class map
    if os.path.exists(args.class_map):
        class_map = get_class_map(args.class_map)
        class_map = {v: int(k) for k, v in class_map.items()}
    else:
        print(f"ERROR: Class Map file provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    try:
        # Load the model with custom metrics
        sam_predictor = get_sam_predictor(args.model_type, device)
        print(f"NOTE: Loaded model {args.model_type}")

    except Exception as e:
        print(f"ERROR: There was an issue loading the model\n{e}")
        sys.exit(1)

    # Setting output variables
    output_dir = args.output_dir
    mask_dir = f"{args.output_dir}masks/"
    os.makedirs(mask_dir, exist_ok=True)

    # Output for mask dataframe
    mask_file = f"{output_dir}masks.csv"

    # Patch size
    patch_size = args.patch_size

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    print("\n###############################################")
    print("Making Masks")
    print("###############################################\n")

    # Subset the images list to only contain those with points
    images = [i for i in images if os.path.basename(i) in points['Name'].unique()]

    # For plotting colored points and masks
    unique_labels = list(class_map.keys())
    label_colors = {l: cmap(i) for i, l in enumerate(unique_labels)}

    # Loop through each image, extract the corresponding patches
    for i_idx, image_path in enumerate(images):

        # Get the points associated
        name = os.path.basename(image_path)
        current_points = points[points['Name'] == name]
        point_colors = current_points['Label'].map(label_colors).values

        # Skip if there are no points for some reason
        if current_points.empty:
            continue

        # Read the image, get the points, create bounding boxes
        image = imread(image_path)

        print(f"NOTE: Making predictions for {name}")
        # Set the image in sam predictor
        sam_predictor.set_image(image)

        # To hold all the updated and colored masks (for viewing)
        updated_mask = np.full(shape=image.shape[:2], fill_value=255)

        # Get all the bounding boxes
        bboxes = []

        for i, r in current_points.iterrows():
            bboxes.append(get_bbox(image, r['Row'], r['Column'], patch_size))

        # Create into a tensor
        bboxes = np.array(bboxes)
        transformed_boxes = torch.tensor(bboxes, device=sam_predictor.device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(transformed_boxes, image.shape[:2])

        # Loop through each point, get bboxes
        for p_idx, (i, r) in tqdm(enumerate(current_points.iterrows())):

            # Every N points
            if p_idx % 10 != 0:
                continue

            # After setting the current image, get masks for each point
            mask, _, _ = sam_predictor.predict_torch(point_coords=None,
                                                     point_labels=None,
                                                     boxes=transformed_boxes[p_idx].unsqueeze(0),
                                                     multimask_output=False)

            # Numpy array masks
            mask = mask.cpu().detach().numpy().astype(np.uint8).squeeze()

            try:
                # Find the most common label within the binary mask (1)
                label = find_most_common_label_in_area(current_points, mask, bboxes[p_idx])
                # convert binary values to correspond to label values
                updated_mask[mask == 1] = int(class_map[label])
            except:
                pass

            if p_idx % 1000 == 0:
                # Colorize the updated mask
                mask_color = colorize_mask(updated_mask, class_map, label_colors)
                # Plot and save the mask
                fname = f"{str(p_idx)}{str(patch_size)}_{name}"
                plot_mask(image, mask_color, points, point_colors, fname, mask_dir)

        # Get the final colored mask, change no data to black
        final_color = colorize_mask(updated_mask, class_map, label_colors)
        final_color[updated_mask == 255, :] = [0, 0, 0]

        # Plot the final mask
        fname = f"final_{str(patch_size)}_{name}"
        plot_mask(image, final_color, points, point_colors, fname, mask_dir)

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

    parser.add_argument("--patch_size", type=int, default=360,
                        help="The approximate size of each superpixel formed by SAM")

    parser.add_argument("--model_type", type=str, default='vit_l',
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
