import os
import sys
import glob
import json
import warnings
import argparse
import requests
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.io import imsave

import torch
import torchvision

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

from Common import log
from Common import get_now
from Common import print_progress
from Common import CACHE_DIR
from Common import IMG_FORMATS

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def download_checkpoint(url, path):
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
            log(f"NOTE: Downloaded file successfully")
            log(f"NOTE: Saved file to {path}")
        else:
            log(f"ERROR: Failed to download file. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        log(f"ERROR: An error occurred: {e}")


def get_sam_predictor(model_type="vit_l", device='cpu', points_per_side=64, points_per_batch=64):
    """

    """
    # URL to download pre-trained weights
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/"

    # The path containing the weights
    sam_root = f"{CACHE_DIR}\\SAM_Weights"
    os.makedirs(sam_root, exist_ok=True)

    # Mapping between the model type, and the checkpoint file name
    sam_dict = {"vit_b": "sam_vit_b_01ec64.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_h": "sam_vit_h_4b8939.pth"}

    if model_type not in list(sam_dict.keys()):
        log(f"ERROR: Invalid model type provided; choices are:\n{list(sam_dict.keys())}")
        sys.exit(1)

    # Checkpoint path to model
    path = f"{sam_root}\\{sam_dict[model_type]}"

    # Check to see if the weights of the model type were already downloaded
    if not os.path.exists(path):
        log("NOTE: Model checkpoint does not exist; downloading")
        url = f"{sam_url}{sam_dict[model_type]}"
        # Download the file
        download_checkpoint(url, path)

    # Loading the mode, returning the predictor
    sam_model = sam_model_registry[model_type](checkpoint=path)
    sam_model.to(device=device)
    sam_predictor = SamAutomaticMaskGenerator(sam_model,
                                              points_per_side=points_per_side,
                                              points_per_batch=points_per_batch)

    return sam_predictor


def find_most_common_label_in_area(points, binary_mask):
    """

    """

    # Filter points within the area
    x = points['Column'].values
    y = points['Row'].values
    l = points['Label'].values

    points_in_area = np.where(binary_mask[y, x])

    # If points land in area, get the most common label
    if points_in_area[0].size > 0:
        labels_in_area = l[points_in_area]
        labels, counts = np.unique(labels_in_area, return_counts=True)
        most_common_label = labels[np.argmax(counts)]
    else:
        # Background
        most_common_label = None

    return most_common_label


def get_exclusive_mask(mask_id, masks):
    """

    """
    # Get the original mask
    result = masks[mask_id]['segmentation'].copy()

    # Loop through all other masks
    for m_idx, m in enumerate(masks):

        if m_idx == mask_id:
            continue

        # Subtract the mask against all others
        exclusive = np.bitwise_xor(result, m['segmentation'].copy())

        # Only retain if there was subtraction
        if exclusive.sum() < result.sum():
            result = exclusive

    return result


def get_color_map(N):
    """

    """
    # Calculate angle intervals given number of classes
    angle_step = 360.0 / N
    angles = [angle_step * i for i in range(N)]

    # For each angle interval, calculate a color in RGB space
    # that maximizes distance from one class to another
    color_coordinates = []

    for angle in angles:
        r = int(255 * (1 + np.cos(np.radians(angle))) / 2)
        g = int(255 * (1 + np.cos(np.radians(angle + 120))) / 2)
        b = int(255 * (1 + np.cos(np.radians(angle + 240))) / 2)
        color_coordinates.append([r, g, b])

    return np.array(color_coordinates)


def colorize_mask(mask, class_map, label_colors):
    """

    """
    # Initialize the RGB mask with zeros
    height, width = mask.shape
    rgb_mask = np.full((height, width, 3), fill_value=255, dtype=np.uint8)

    # dict with index as key, rgb as value
    cmap = {v: label_colors[k][0:3] for k, v in class_map.items()}

    # Loop through all index values
    # Set rgb color in colored mask
    for val in np.unique(mask):
        if val in class_map.values():
            color = np.array(cmap[val]) * 255
            rgb_mask[mask == val, :] = color.astype(np.uint8)

    return rgb_mask.astype(np.uint8)


def plot_mask(image, mask_color, points, point_colors, plot_path):
    """

    """

    # Plot title
    fname = os.path.basename(plot_path)

    # Plot masks
    plt.figure(figsize=(10, 10))
    plt.title(fname)
    plt.imshow(image)
    plt.imshow(mask_color, alpha=.75)
    plt.scatter(points['Column'].values, points['Row'].values, c=point_colors, s=1)

    # Save the plot
    plt.savefig(plot_path)
    plt.close()


def mss(args):
    """

    """
    log("\n###############################################")
    log("Multilevel Superpixel Segmentation w/ SAM")
    log("###############################################\n")

    # Check for CUDA
    log(f"NOTE: PyTorch version - {torch.__version__}")
    log(f"NOTE: Torchvision version - {torchvision.__version__}")
    log(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert confidence value to float
    confidence = float(args.confidence / 100)

    if not 0 <= confidence <= 1.0:
        log(f"ERROR: Confidence value must be between [0 - 100]")
        sys.exit(1)

    # Predictions Dataframe
    if os.path.exists(args.annotations):
        points_df = pd.read_csv(args.annotations, index_col=0)
        image_names = np.unique(points_df['Name'].to_numpy())

        # Create class map and color map based on annotation file
        if args.label_col in points_df.columns:
            label_col = args.label_col
            log(f"NOTE: Using labels from the column '{label_col}'")
        else:
            log(f"ERROR: Column {args.label_col} doesn't exist in {args.annotations}")
            sys.exit(1)

        # Filter based on confidence scores of the label colum;
        # if they use 'Label' deal with it
        if 'suggestion' in label_col:
            points_df = points_df[points_df[label_col.replace("suggestion", "confidence")] >= confidence]

        # Make a subset containing only necessary fields
        points_df = points_df[['Name', 'Row', 'Column', label_col]]
        points_df.columns = ['Name', 'Row', 'Column', 'Label']

        log(f"NOTE: Found a total of {len(points_df)} sampled points for {len(image_names)} images")

        # Create the class mapping between values and colors
        class_map = {l: i + 1 for i, l in enumerate(sorted(points_df['Label'].unique()))}

        # Create a color map give the amount of classes
        unique_labels = list(class_map.keys())
        color_map = get_color_map(len(unique_labels))
        # Get the colors per class
        label_colors = {l: color_map[i] / 255.0 for i, l in enumerate(unique_labels)}

    else:
        log("ERROR: Points file provided doesn't exist.")
        sys.exit(1)

    # Image files
    if os.path.exists(args.images):
        # Images that are the correct image format
        images = [i for i in glob.glob(f"{args.images}/*.*") if i.split(".")[-1].lower() in IMG_FORMATS]
        # Subset the images list to only contain those with points
        images = [i for i in images if os.path.basename(i) in image_names]
        if not images:
            raise Exception(f"ERROR: No images were found in the directory provided; please check input.")
        else:
            log(f"NOTE: Found {len(images)} images in directory provided")
    else:
        log("ERROR: Image directory provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    try:
        # Load the model with custom metrics
        sam_predictor = get_sam_predictor(args.model_type, device)
        log(f"NOTE: Loaded model {args.model_type}")

    except Exception as e:
        log(f"ERROR: There was an issue loading the model\n{e}")
        sys.exit(1)

    # Setting output directories
    output_dir = f"{args.output_dir}\\masks\\SAM_{get_now()}\\"
    plot_dir = f"{output_dir}\\plots\\"
    seg_dir = f"{output_dir}\\segs\\"
    color_dir = f"{output_dir}\\color\\"
    overlay_dir = f"{output_dir}\\overlay\\"

    output_mask_csv = f"{output_dir}masks.csv"
    output_color_json = f"{output_dir}Color_Map.json"

    # Create the output directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Output for mask dataframe
    mask_df = []

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    log("\n###############################################")
    log("Making Masks")
    log("###############################################\n")

    # Loop through each image, extract the corresponding patches
    for i_idx, image_path in enumerate(images):

        # Get the points associated with current image
        name = os.path.basename(image_path)
        current_points = points_df[points_df['Name'] == name]

        # Skip if there are no points for some reason
        if current_points.empty:
            continue

        # Read the image, get the points, create bounding boxes
        image = imread(image_path)
        height, width = image.shape[0:2]
        area = height * width

        log(f"NOTE: Making predictions for {name}")
        # Set the image in sam predictor
        masks = sam_predictor.generate(image)
        # Sort based on area (larger first)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        # To hold the updated mask, will be added onto each iteration
        final_mask = np.full(image.shape[:2], fill_value=0, dtype=np.uint8)

        # Loop through all masks generated
        for m_idx in range(len(masks)):

            # Get the generated mask
            mask = masks[m_idx]['segmentation']

            # Check the area; if it's large, subtract it
            # from the other masks so there isn't overlap
            if masks[m_idx]['area'] / area > 0.35:
                mask = get_exclusive_mask(m_idx, masks)

            # Get the most common label in the mask area
            label = find_most_common_label_in_area(current_points, mask)

            # If it's not background
            if label:
                label = int(class_map[label])
                final_mask[mask] = label

        # ------------------------------------------------
        # Save the final masks
        # ------------------------------------------------

        # Save the seg mask
        mask_path = f"{seg_dir}{name.split('.')[0]}.png"
        imsave(fname=mask_path, arr=final_mask.astype(np.uint8))
        log(f"NOTE: Saved seg mask to {mask_path}")

        # Get the final colored mask, change no data to black
        final_color = colorize_mask(final_mask, class_map, label_colors)
        final_color[final_mask == 255, :] = [0, 0, 0]

        # Save the color mask
        color_path = f"{color_dir}{name.split('.')[0]}.png"
        imsave(fname=color_path, arr=final_color.astype(np.uint8))
        log(f"NOTE: Saved color mask to {color_path}")

        # Get the final overlay, which is the final color mask
        # on top of the original image, with 50% transparency, while
        # maintaining the original resolution
        final_overlay = cv2.addWeighted(image, 0.5, final_color, 0.5, 0)

        # Save the overlay
        overlay_path = f"{overlay_dir}{name.split('.')[0]}.png"
        imsave(fname=overlay_path, arr=final_overlay.astype(np.uint8))
        log(f"NOTE: Saved overlay to {overlay_path}")

        # Add to output list
        mask_df.append([name, image_path, mask_path, color_path, overlay_path])

        if args.plot:
            # Get the point colors to plot
            point_colors = current_points[label_col].map(label_colors).values
            # Plot the final mask
            fname = f"{name.split('.')[0]}.jpg"
            plot_path = f"{plot_dir}{fname}"
            plot_mask(image, final_color, current_points, point_colors, plot_path)
            log(f"NOTE: Saved plot to {plot_path}")

        # Gooey
        print_progress(i_idx, len(image_names))

    # Save dataframe to root directory
    mask_df = pd.DataFrame(mask_df, columns=['Name', 'Image Path', 'Seg Path', 'Color Path', 'Overlay Path'])
    mask_df.to_csv(output_mask_csv)

    if os.path.exists(output_mask_csv):
        log(f"NOTE: Mask dataframe saved to {output_dir}")
    else:
        log(f"ERROR: Could not save mask dataframe")

    # Create a final class mapping for the seg masks
    seg_map = {k: {} for k in class_map.keys()}

    for l in class_map.keys():
        seg_map[l]['id'] = class_map[l]
        seg_map[l]['color'] = (np.array(label_colors[l][0:3]) * 255).astype(np.uint8).tolist()

    # Save the color mapping json file
    with open(output_color_json, 'w') as output_file:
        json.dump(seg_map, output_file, indent=4)

    if os.path.exists(output_color_json):
        log(f"NOTE: Color Mapping JSON file saved to {output_dir}")
    else:
        log(f"ERROR: Could not save Color Mapping JSON file")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Multilevel Superpixel Segmentation w/ SAM")

    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to the points file containing 'Name', 'Row', 'Column', and 'Label' information.")

    parser.add_argument("--label_col", type=str, required=True,
                        help="The column in annotations with labels to use ('Label', 'Machine suggestion N, etc).")

    parser.add_argument("--confidence", type=int, default=75,
                        help="Confidence threshold value to filter")

    parser.add_argument("--model_type", type=str, default='vit_b',
                        help="Model to use; one of ['vit_b', 'vit_l', 'vit_h']")

    parser.add_argument("--plot", action='store_true',
                        help="Saves figures of final masks")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        mss(args)
        log("Done.\n")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()
