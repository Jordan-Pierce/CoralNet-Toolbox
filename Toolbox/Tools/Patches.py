import warnings
warnings.filterwarnings("ignore")

import os
import glob
import argparse
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread, imsave

from Toolbox.Tools import *


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


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

    # Top of the patch, else edge of image
    top = y - size
    if top < 0:
        top_pad = abs(top)
        top = 0

    # Bottom of patch, else edge of image
    bottom = y + size
    if bottom > height:
        bottom_pad = bottom - height
        bottom = height

    # Left of patch, else edge of image
    left = x - size
    if left < 0:
        left_pad = abs(left)
        left = 0

    # Right of patch, else edge of image
    right = x + size
    if size > width:
        right_pad = right - width
        right = width

    # Get the sub-image from image
    patch = image[top: bottom, left: right, :]

    # Check if the sub-image size is smaller than N x N
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        # Pad the sub-image with zeros if it's along the border
        patch = np.pad(patch, ((top_pad, bottom_pad),
                               (left_pad, right_pad),
                               (0, 0)))

    # Check if the percentage of padded zeros is more than 50%
    if np.mean(patch == 0) > 0.75:
        patch = None

    # If for some reason it's not the right size, ignore (fixed this)
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = None

    return patch


def process_image(image_name, image_dir, annotation_df, output_dir):
    # Get the name and path
    image_prefix = image_name.split(".")[0]
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"ERROR: Image {image_path} does not exist; skipping")
        return

    # Open the image as np array just once
    image = imread(image_path)
    # Get the annotations specific to this image
    image_df = annotation_df[annotation_df['Name'] == image_name]

    # List to hold patches
    patches = []

    # Loop through each annotation for this image
    for i, r in image_df.iterrows():
        try:
            # Extract the patch
            patch = crop_patch(image, r['Row'], r['Column'])
            name = f"{image_prefix}_{r['Row']}_{r['Column']}_{r['Label']}.png"
            path = os.path.join(output_dir, 'patches', r['Label'], name)

            # If it's not mostly empty, crop it
            if patch is not None:
                # Save the patch
                imsave(fname=path, arr=patch)
                # Add to list
                patches.append([name, path, r['Label'], image_name, image_path])

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return patches


def crop_patches(annotation_file, image_dir, output_dir):
    """
    Given an image dataframe, this function will crop a patch for each annotation
    """

    print("\n###############################################")
    print("Cropping Patches")
    print("###############################################\n")

    if os.path.exists(annotation_file):
        annotation_df = pd.read_csv(annotation_file)
    else:
        raise Exception(f"ERROR: Patch_Extractor file {annotation_file} does not exist")

    # Make sub-folders for all the class categories
    for label in annotation_df['Label'].unique():
        os.makedirs(os.path.join(output_dir, 'patches', label), exist_ok=True)

    # All unique images in the annotation dataframe
    image_names = annotation_df['Name'].unique()

    # All patches
    patches = []

    # For gooey
    prg_total = len(annotation_df)

    # Using ThreadPoolExecutor to process each image concurrently
    with ThreadPoolExecutor() as executor:
        future_to_patches = {
            executor.submit(process_image, image_name, image_dir, annotation_df, output_dir): image_name
            for image_name in image_names
        }

        for future in concurrent.futures.as_completed(future_to_patches):
            image_name = future_to_patches[future]
            patches.extend(future.result())
            print_progress(len(patches), prg_total)

    # Save patches dataframe
    patches_path = os.path.join(output_dir, 'patches.csv')
    patches_df = pd.DataFrame(patches, columns=['Name', 'Path', 'Label', 'Image Name', 'Image Path'])
    patches_df.to_csv(patches_path)

    if os.path.exists(patches_path):
        print(f"NOTE: Patches dataframe saved to {patches_path}")
    else:
        print(f"ERROR: Patches dataframe could not be saved")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='Patch arguments')

    parser.add_argument('--annotation_file', type=str, nargs="+", default=[],
                        help='The path to annotation file(s); expects CoralNet format')

    parser.add_argument('--output_dir', type=str, default=os.path.abspath("../../Data"),
                        help='A root directory where all output will be saved to.')

    args = parser.parse_args()

    try:

        crop_patches(args.annotation_file, args.output_dir)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
