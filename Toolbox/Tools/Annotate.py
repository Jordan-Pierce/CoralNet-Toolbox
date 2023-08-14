import os
import sys
import shutil
import subprocess

import re
import time
import glob
import datetime
import argparse

import pandas as pd
from PIL import Image

from Toolbox.Tools import *


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def print_labelset(labelset):
    """
    Pretty prints to user in icons the total number of annotations made in session
    """
    print("\n" * 20)
    print("\n###############################################")
    print("Label Tracker")
    print("###############################################\n")

    for k, v in labelset.items():
        if v:
            print(f"{k}: {v}", flush=True)


def extract_label(string):
    """
    Pulls the label from the patch name in the log file
    """

    # Anything before "_LETTER_.bmp"
    pattern = r'^([^_]+(?:_[^_]+)*?)_[a-zA-Z]_[^_]+\.bmp$'

    match = re.match(pattern, string)

    if match:
        return match.group(1)
    else:
        return None


def app_still_on(process):
    """
    A way to determine if the application is still on
    """

    app_on = True

    try:
        # Check to see if app is on, catch the error if it is
        process.communicate("", .1)
        # If app is off, no error is thrown
        app_on = False

    except:
        # App is still on, continue moving patches to tmp dir
        pass

    return app_on


def annotate(args):
    """
    Opens the patch extractor tool via subprocess, and continuously logs results while app is still running.
    Once it's closed, the results are stored as annotation dataframe in root directory, and temp files are deleted.
    """

    print("\n###############################################")
    print("Patch Extractor Tool")
    print("###############################################\n")

    if not os.path.exists(args.image_dir):
        raise NotADirectoryError(f"ERROR: Image Directory not found. Please check the path provided.")

    if not os.path.exists(args.patch_extractor_path):
        raise FileNotFoundError(f"ERROR: Executable not found. Please check the path provided.")

    # Set the image directory
    if os.path.exists(args.image_dir):
        image_dir = args.image_dir
    else:
        print(f"ERROR: Image directory provided doesn't exist; please check input")
        sys.exit(1)

    # Set exe
    if os.path.exists(args.patch_extractor_path):
        exe_path = args.patch_extractor_path
    else:
        print(f"ERROR: Path to Patch Extractor doesn't exist; please check input")
        sys.exit(1)

    # Set the paths for output
    output_dir = f"{os.path.dirname(image_dir)}\\annotations\\"
    output_path = f"{output_dir}{get_now()}_annotations.csv"
    os.makedirs(output_dir, exist_ok=True)
    # Create a directory for temp patches
    tmp_dir = f"{image_dir}/tmp/"
    os.makedirs(tmp_dir, exist_ok=True)

    # Keeping track of labels made
    labelset = {}

    # Run the application in the background
    process = subprocess.Popen(['start', '', exe_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True)

    # Determine the current log file
    log_files = list(set(glob.glob(f"{image_dir}/CNNDataExtractor-*.txt")))
    num_logs = len(log_files)
    log_file = None

    # The log file will be created after a single patch is extracted
    while app_still_on(process):
        # Looks for log files
        log_files = list(set(glob.glob(f"{image_dir}/CNNDataExtractor-*.txt")))
        # If there is a new one, set it as the log file for this session
        if num_logs != len(log_files):
            # The log file name will stay the same throughout the entire session
            log_file = max(log_files, key=os.path.getmtime)
            print(f"NOTE: Logging to {os.path.basename(log_file)}")
            break

    # Continuously loop while app is on
    while app_still_on(process):

        # Grab the unique patches, and log files
        patch_files = list(set(glob.glob(f"{image_dir}/*.bmp")))

        # Move all the patch files to the tmp directory behind the scenes
        for patch_file in patch_files:
            patch_name = os.path.basename(patch_file)
            dst = f"{tmp_dir}{patch_name}"
            shutil.move(patch_file, dst)
            # Print to user
            extracted_label = extract_label(patch_name)
            if extracted_label is None:
                print("WARNING: Annotation not recorded; convention should only contain a single '_'")
            else:
                print(f"NOTE: Annotation created for class '{extracted_label}'")
                if extracted_label not in labelset:
                    labelset[extracted_label] = 0

                # Add to the distribution
                labelset[extracted_label] += 1
                # Pretty print to user
                print_labelset(labelset)

    # App has been closed, now process the annotations
    if log_file:
        dataframe = pd.read_csv(log_file, sep='\t')
        dataframe.columns = ['Image Path', 'Column', 'Row', 'Name']
        # Extract labels, store as column
        dataframe['Label'] = [extract_label(n) for n in dataframe['Name'].values]
        dataframe['Image Name'] = [os.path.basename(n) for n in dataframe['Image Path'].values]

        # Will hold image 'Name', Row, Column, and Label
        annotations = []

        for i, r in dataframe.iterrows():

            # Get size, adjust Row, Column
            patch_path = f"{tmp_dir}/{r['Name']}"

            if not os.path.exists(patch_path):
                print("WARNING: Do not move patches while still annotating!")
                continue

            # Width and height of patch
            w, h = Image.open(patch_path).size

            # Updating column and row
            column = r['Column'] + (h // 2)
            row = r['Row'] + (w // 2)

            image_name = r['Image Name']
            label = r['Label']

            # Add to annotations
            annotations.append([image_name, row, column, label])

        # Save as dataframe to root directory
        annotations = pd.DataFrame(annotations, columns=['Name', 'Row', 'Column', 'Label'])
        # Save, and make sure it exists
        annotations.to_csv(output_path)

        if os.path.exists(output_path):
            print(f"\nNOTE: Annotations saved in {output_path}")

        else:
            print(f"ERROR: Could not save annotations; leaving tmp files where they are.")
            sys.exit(1)

        # Clean up
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

        if os.path.exists(log_file) and os.path.isfile(log_file):
            os.remove(log_file)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description='API arguments')

    parser.add_argument('--patch_extractor_path', type=str,
                        default=os.path.abspath('./Patch_Extractor/CNNDataExtractor.exe'))

    parser.add_argument('--image_dir', required=True, type=str,
                        help='A directory where all images are located.')


    args = parser.parse_args()

    try:
        # Call the annotate function
        annotate(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
