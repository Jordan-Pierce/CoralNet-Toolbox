import os
import argparse
import warnings
import traceback

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from src.Common import get_now
from src.Common import console_user

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def crop_patch(image, y, x, patch_size=224):
    """
    Given an image, and a Y, X location, this function will extract the patch.
    """

    height, width, _ = image.shape
    x = int(x)
    y = int(y)

    # N x N
    size = patch_size // 2

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
    if right > width:
        right_pad = right - width
        right = width

    # Get the sub-image from image
    patch = image[top: bottom, left: right, :]

    # Check if the sub-image size is smaller than N x N
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        # Pad the sub-image with zeros if it's along the border
        patch = np.pad(patch, ((top_pad, bottom_pad),
                               (left_pad, right_pad),
                               (0, 0)))

    # Resize the patch to 224 no matter what
    patch = (resize(patch, (224, 224)) * 255).astype(np.uint8)[:, :, 0:3]

    return patch


def process_image(image_name, image_dir, annotation_df, output_dir, patch_size):
    """

    """
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
            patch = crop_patch(image, r['Row'], r['Column'], patch_size)
            name = f"{image_prefix}_{r['Row']}_{r['Column']}_{r['Label']}.jpg"
            path = os.path.join(output_dir, 'patches', r['Label'], name)

            # Save
            if patch is not None:
                # Save the patch
                imsave(fname=path, arr=patch, quality=90)
                # Add to list
                patches.append([name, path, r['Label'], r['Row'], r['Column'], image_name, image_path])

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return patches


def patches(args):
    """
    Given an image dataframe, this function will crop a patch for each annotation
    """

    print("\n###############################################")
    print("Cropping Patches")
    print("###############################################\n")

    if os.path.exists(args.image_dir):
        image_dir = args.image_dir
    else:
        raise Exception(f"ERROR: Image directory provided doesn't exist; please check input")

    if os.path.exists(args.annotation_file):
        annotation_file = args.annotation_file
        annotation_df = pd.read_csv(annotation_file)

        assert "Row" in annotation_df.columns, print(f"ERROR: 'Row' not in provided csv")
        assert "Column" in annotation_df.columns, print(f"ERROR: 'Column' not in provided csv")
        assert args.label_column in annotation_df.columns, print(f"ERROR: '{args.label_column}' not in provided csv")
        assert args.image_column in annotation_df.columns, print(f"ERROR: {args.image_column} not in provided csv")
    else:
        raise Exception(f"ERROR: Annotation file provided does not exist; please check input")

    # Create output
    output_dir = f"{args.output_dir}/patches/{get_now()}/"
    output_path = f"{output_dir}patches.csv"
    os.makedirs(output_dir, exist_ok=True)

    # Make sub-folders for all the class categories
    for label in annotation_df[args.label_column].unique():
        os.makedirs(os.path.join(output_dir, 'patches', label), exist_ok=True)

    # Subset of annotation df, simpler
    sub_df = annotation_df[[args.image_column, 'Row', 'Column', args.label_column]]
    sub_df.columns = ['Name', 'Row', 'Column', 'Label']
    # All unique images in the annotation dataframe
    image_names = sub_df['Name'].unique()

    # Size of patches to crop
    patch_size = args.patch_size

    # All patches
    patches = []

    # Using ThreadPoolExecutor to process each image concurrently
    with ThreadPoolExecutor() as executor:
        future_to_patches = {
            executor.submit(process_image, image_name, image_dir, sub_df, output_dir, patch_size): image_name
            for image_name in image_names
        }

        for future in concurrent.futures.as_completed(future_to_patches):
            image_name = future_to_patches[future]
            patches.extend(future.result())

    # Save patches dataframe
    patches = pd.DataFrame(patches, columns=['Name', 'Path',
                                             'Label', 'Row', 'Column',
                                             'Image Name', 'Image Path'])
    # Save as CSV
    patches.to_csv(output_path)

    if os.path.exists(output_path):
        print(f"NOTE: Patches dataframe saved to {output_path}")
        return output_path
    else:
        print(f"ERROR: Patches dataframe could not be saved")
        return None


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='Patch arguments')

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory with images associated with annotation file')

    parser.add_argument('--annotation_file', type=str, nargs="+", default=[],
                        help='The path to annotation file(s); expects CoralNet format')

    parser.add_argument("--image_column", type=str, default="Name",
                        help="The column specifying the image basename")

    parser.add_argument("--label_column", required=False, type=str, default='Label',
                        help='Label column in Annotations dataframe.')

    parser.add_argument("--patch_size", type=int, default=112,
                        help="The size of each patch extracted")

    parser.add_argument('--output_dir', type=str, required=True,
                        help='A root directory where all output will be saved to.')

    args = parser.parse_args()

    try:

        patches(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()