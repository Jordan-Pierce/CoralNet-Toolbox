import os
import sys
import glob
import argparse
import traceback

import random
import numpy as np
import pandas as pd
from PIL import Image

from Common import IMG_FORMATS
from Common import get_now
from Common import print_progress


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def points(args):
    """
    Generates a set of sample coordinates within a given image size.
    """

    print("\n###############################################", flush=True)
    print("Sample Points", flush=True)
    print("###############################################\n", flush=True)

    # Set the variables
    sample_method = args.sample_method
    num_points = args.num_points

    if os.path.exists(args.images):
        images = args.images
        image_files = [i for i in glob.glob(f'{images}\*.*') if os.path.exists(i)]
        image_files = [i for i in image_files if os.path.basename(i).split(".")[-1].lower() in IMG_FORMATS]

        if not image_files:
            print("ERROR: No image files found in directory provided; please check input", flush=True)
            sys.exit(1)

        print(f"NOTE: Sampling {num_points} points for {len(image_files)} images", flush=True)
    else:
        print(f"ERROR: Image directory provided doesn't exist; please check provided input", flush=True)
        sys.exit(1)

    # Create output
    output_dir = f"{args.output_dir}\\points\\"
    output_file = f"{output_dir}{get_now()}_points.csv"
    os.makedirs(output_dir, exist_ok=True)

    samples = []

    for idx, image_file in enumerate(image_files):

        # Gooey
        print_progress(idx, len(image_files))

        # Read it to get the size
        image_name = os.path.basename(image_file)
        img = Image.open(image_file)
        width, height = img.size

        # At least half patch
        min_width = 0 + 112
        max_width = width - 112
        min_height = 0 + 112
        max_height = height - 112

        # Generate Uniform Samples
        if sample_method == 'Uniform':
            x_coords = np.linspace(min_width, max_width - 1, int(np.sqrt(num_points)))
            y_coords = np.linspace(min_height, max_height - 1, int(np.sqrt(num_points)))
            for x in x_coords:
                for y in y_coords:
                    samples.append({'Name': image_name,
                                    'Row': int(y),
                                    'Column': int(x),
                                    'Label': 'Unlabeled'})

        # Generate Random Samples
        elif sample_method == 'Random':
            for i in range(num_points):
                x = random.randint(min_width, max_width - 1)
                y = random.randint(min_height, max_height - 1)
                samples.append({'Name': image_name,
                                'Row': int(y),
                                'Column': int(x),
                                'Label': 'Unlabeled'})

        # Generate Stratified Samples
        else:
            n = int(np.sqrt(num_points))
            x_range = np.linspace(min_width, max_width - 1, n + 1)
            y_range = np.linspace(min_height, max_height - 1, n + 1)
            for i in range(n):
                for j in range(n):
                    x = np.random.uniform(x_range[i], x_range[i + 1])
                    y = np.random.uniform(y_range[j], y_range[j + 1])
                    samples.append({'Name': image_name,
                                    'Row': int(y),
                                    'Column': int(x),
                                    'Label': 'Unlabeled'})
    if samples:
        print(f"NOTE: Saving {len(samples)} sampled points", flush=True)
        # Store as dataframe
        samples = pd.DataFrame.from_records(samples)
        # Save as csv to points folder
        samples.to_csv(output_file)

        if os.path.exists(output_file):
            print(f"NOTE: Sampled points saved to {output_file}", flush=True)
        else:
            print(f"ERROR: Sampled points could not be saved; check provided input.", flush=True)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='Patch arguments')

    parser.add_argument('--images', type=str,
                        help='Directory of images to create sampled points for')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='A root directory where all output will be saved to.')

    parser.add_argument('--sample_method', type=str,
                        help='Method used to sample points from each image [Uniform, Random, Stratified]')

    parser.add_argument('--num_points', type=int, default=200,
                        help='The number of points to sample from each image')

    args = parser.parse_args()

    try:

        points(args)
        print("Done.", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
