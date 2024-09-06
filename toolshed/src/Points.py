import os
import sys
import glob
import argparse
import traceback

import random
import numpy as np
import pandas as pd
from PIL import Image

from src.Common import get_now
from src.Common import console_user
from src.Common import IMG_FORMATS
from src.Common import progress_printer


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def get_points(min_width, min_height, max_width, max_height, sample_method='Random', num_points=200):
    """

    """
    x = []
    y = []

    # Generate Uniform Samples # TODO This throws an error
    if sample_method == 'Uniform':
        x_coords = np.linspace(min_width, max_width - 1, int(np.sqrt(num_points))).astype(int)
        y_coords = np.linspace(min_height, max_height - 1, int(np.sqrt(num_points))).astype(int)
        for x_coord in x_coords:
            for y_coord in y_coords:
                x.append(x_coord)
                y.append(y_coord)

    # Generate Random Samples
    elif sample_method == 'Random':
        for i in range(num_points):
            x_coord = random.randint(min_width, max_width - 1)
            y_coord = random.randint(min_height, max_height - 1)

            x.append(x_coord)
            y.append(y_coord)

    # Generate Stratified Samples
    else:
        n = int(np.sqrt(num_points))
        x_range = np.linspace(min_width, max_width - 1, n + 1)
        y_range = np.linspace(min_height, max_height - 1, n + 1)
        for i in range(n):
            for j in range(n):
                x_coords = np.random.uniform(x_range[i], x_range[i + 1])
                y_coords = np.random.uniform(y_range[j], y_range[j + 1])

                x.append(int(x_coords))
                y.append(int(y_coords))

    return np.array(x), np.array(y)


def points(args):
    """
    Generates a set of sample coordinates within a given image size.
    """

    print("\n###############################################")
    print("Sample Points")
    print("###############################################\n")

    # Set the variables
    sample_method = args.sample_method
    num_points = args.num_points

    if os.path.exists(args.images):
        images = args.images
        image_files = [i for i in glob.glob(f'{images}\*.*') if os.path.exists(i)]
        image_files = [i for i in image_files if os.path.basename(i).split(".")[-1].lower() in IMG_FORMATS]

        if not image_files:
            print("ERROR: No image files found in directory provided; please check input")
            sys.exit(1)

        print(f"NOTE: Sampling {num_points} points for {len(image_files)} images")
    else:
        print(f"ERROR: Image directory provided doesn't exist; please check provided input")
        sys.exit(1)

    # Create output
    output_dir = f"{args.output_dir}/points/"
    output_file = f"{output_dir}{get_now()}_points.csv"
    os.makedirs(output_dir, exist_ok=True)

    samples = []

    for idx, image_file in progress_printer(enumerate(image_files)):

        # Read it to get the size
        image_name = os.path.basename(image_file)
        img = Image.open(image_file)
        width, height = img.size

        # Slight offset
        min_width = 0 + 32
        max_width = width - 32
        min_height = 0 + 32
        max_height = height - 32

        x, y = get_points(min_width, min_height, max_width, max_height, sample_method, num_points)

        for _ in range(len(list(zip(x, y)))):

            samples.append({'Name': image_name,
                            'Row': int(y[_]),
                            'Column': int(x[_]),
                            'Label': 'Unlabeled'})

    if samples:
        print(f"NOTE: Saving {len(samples)} sampled points")
        # Store as dataframe
        samples = pd.DataFrame.from_records(samples)
        # Save as csv to points folder
        samples.to_csv(output_file)

        if os.path.exists(output_file):
            print(f"NOTE: Sampled points saved to {output_file}")
        else:
            print(f"ERROR: Sampled points could not be saved; check provided input.")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='Patch arguments')

    parser.add_argument('--images', type=str,
                        help='Directory of images to create sampled points for')

    parser.add_argument('--sample_method', type=str,
                        help='Method used to sample points from each image [Uniform, Random, Stratified]')

    parser.add_argument('--num_points', type=int, default=200,
                        help='The number of points to sample from each image')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='A root directory where all output will be saved to.')

    args = parser.parse_args()

    try:

        points(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()