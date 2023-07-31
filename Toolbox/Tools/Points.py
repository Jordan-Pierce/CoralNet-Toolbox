import os
import glob

import random
import numpy as np
import pandas as pd
from PIL import Image

from Toolbox.Tools import *


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def points(args):
    """
    Generates a set of sample coordinates within a given image size.
    """

    print("\n###############################################")
    print("Sample Points")
    print("###############################################\n")

    # Set the variables
    image_files = args.image_files
    output_dir = args.output_dir
    sample_method = args.sample_method
    num_points = args.num_points

    if image_files is []:
        print(f"ERROR: No images provided; please check provided input.")
    else:
        image_files = [i for i in image_files if os.path.exists(i)]
        print(f"NOTE: Sampling {num_points} points for {len(image_files)} images")

    # Make sure output directory is there
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/points.csv"

    samples = []

    for idx, image_file in enumerate(image_files):

        # Gooey
        print_progress(idx, len(image_files))

        # Read it to get the size
        image_name = os.path.basename(image_file)
        img = Image.open(image_file)
        width, height = img.size

        # TODO modify this idea so that it's the margins of the image
        # At least half patch
        width -= 112
        height -= 112

        # Generate Uniform Samples
        if sample_method == 'Uniform':
            x_coords = np.linspace(0, width - 1, int(np.sqrt(num_points)))
            y_coords = np.linspace(0, height - 1, int(np.sqrt(num_points)))
            for x in x_coords:
                for y in y_coords:
                    samples.append({'Name': image_name,
                                    'Row': int(y),
                                    'Column': int(x),
                                    'Label': 'Unlabeled'})

        # Generate Random Samples
        elif sample_method == 'Random':
            for i in range(num_points):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                samples.append({'Name': image_name,
                                'Row': int(y),
                                'Column': int(x),
                                'Label': 'Unlabeled'})

        # Generate Stratified Samples
        else:
            n = int(np.sqrt(num_points))
            x_range = np.linspace(0, width - 1, n + 1)
            y_range = np.linspace(0, height - 1, n + 1)
            for i in range(n):
                for j in range(n):
                    x = np.random.uniform(x_range[i], x_range[i + 1])
                    y = np.random.uniform(y_range[j], y_range[j + 1])
                    samples.append({'Name': image_name,
                                    'Row': int(y),
                                    'Column': int(x),
                                    'Label': 'Unlabeled'})
    if samples:

        print(f"NOTE: Saving {len(samples)} sampled points")

        # Store as dataframe
        samples = pd.DataFrame.from_records(samples)

        # Save to dataframe
        if os.path.exists(output_file):
            previous_samples = pd.read_csv(output_file, index_col=0)
            samples = pd.concat((previous_samples, samples))
            samples.drop_duplicates(inplace=True)

        # Save
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

    parser.add_argument('--image_files', type=str, nargs="+",
                        help='Images to create sampled points for')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='A root directory where all output will be saved to.')

    parser.add_argument('--sample_method', type=str,
                        help='Method used to sample points from each image [Uniform, Random, Stratified]')

    parser.add_argument('--num_points', type=int, default=200,
                        help='The number of points to sample from each image')

    args = parser.parse_args()

    try:

        points(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
