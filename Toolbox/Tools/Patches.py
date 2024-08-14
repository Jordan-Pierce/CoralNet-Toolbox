import os
import argparse
import warnings
import traceback

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from Common import get_now
from Common import console_user

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def crop_patch(image, y, x, patch_size=224):
    """
    Given an image, and a Y, X location, this function will extract the patch.

    :param image:
    :param y:
    :param x:
    :param patch_size:
    :return:
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


def process_image(image_name, image_dir, annotations, output_dir, patch_size):
    """

    :param image_name:
    :param image_dir:
    :param annotations:
    :param output_dir:
    :param patch_size:
    :return:
    """
    image_prefix = image_name.split(".")[0]
    image_path = f"{image_dir}/{image_name}"

    if not os.path.exists(image_path):
        print(f"ERROR: Image {image_path} does not exist; skipping")
        return []

    image = imread(image_path)

    patch_data = []

    for annotation in annotations:
        try:
            patch = crop_patch(image, annotation['Row'], annotation['Column'], patch_size)
            name = f"{image_prefix}_{annotation['Row']}_{annotation['Column']}_{annotation['Label']}.jpg"
            path = f"{output_dir}/patches/{annotation['Label']}/{name}"

            if patch is not None:
                imsave(fname=path, arr=patch, quality=90)
                patch_data.append({
                    'Name': name,
                    'Path': path,
                    'Label': annotation['Label'],
                    'Row': annotation['Row'],
                    'Column': annotation['Column'],
                    'Image Name': image_name,
                    'Image Path': image_path
                })

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return patch_data


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
        annotation_df.dropna(inplace=True)

        assert "Name" in annotation_df.columns, print(f"ERROR: 'Name' not in provided csv")
        assert "Row" in annotation_df.columns, print(f"ERROR: 'Row' not in provided csv")
        assert "Column" in annotation_df.columns, print(f"ERROR: 'Column' not in provided csv")
        assert args.label_column in annotation_df.columns, print(f"ERROR: '{args.label_column}' not in provided csv")
        assert args.image_column in annotation_df.columns, print(f"ERROR: {args.image_column} not in provided csv")

        annotation_df['Name'] = [os.path.basename(p) for p in annotation_df['Name'].values]
    else:
        raise Exception(f"ERROR: Annotation file provided does not exist; please check input")

    # Create output
    output_name = args.output_name if args.output_name else get_now()
    output_dir = f"{args.output_dir}/patches/{output_name}"
    output_path = f"{output_dir}/patches.csv"
    os.makedirs(output_dir, exist_ok=True)

    # Create a dictionary of image names to their annotations
    image_annotations = {}
    for _, row in annotation_df.iterrows():
        image_name = os.path.basename(row[args.image_column])

        if image_name not in image_annotations:
            image_annotations[image_name] = []

        image_annotations[image_name].append({
            'Row': row['Row'],
            'Column': row['Column'],
            'Label': row[args.label_column]
        })

    # Make sub-folders for all the class categories
    for label in annotation_df[args.label_column].unique():
        os.makedirs(os.path.join(output_dir, 'patches', label), exist_ok=True)

    # Create an empty list to store patch information
    patches_data = []

    def process_image_wrapper(image_name):
        return process_image(image_name,
                             image_dir,
                             image_annotations[image_name],
                             output_dir,
                             args.patch_size)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image_wrapper, image_annotations.keys())

    for result in results:
        patches_data.extend(result)

    # Create a DataFrame from the collected patch data
    patches_df = pd.DataFrame(patches_data,
                              columns=['Name', 'Path', 'Label', 'Row', 'Column', 'Image Name', 'Image Path'])

    # Save the DataFrame to CSV
    patches_df.to_csv(output_path, index=False)

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

    parser.add_argument('--output_name', type=str, default="",
                        help='A name to give the out output dir; else timestamped.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='A root directory where all output will be saved to.')

    args = parser.parse_args()

    try:

        patches(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()