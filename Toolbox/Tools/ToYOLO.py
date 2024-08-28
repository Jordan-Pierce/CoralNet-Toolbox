import os
import uuid
import json
import random
import shutil
import argparse
import traceback
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from Classification import downsample_majority_classes

from Common import get_now
from Common import console_user


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------


def create_class_mapping(labels, output_dir):
    """

    :param labels:
    :param output_dir:
    :return:
    """
    result = {}

    for label in labels:
        color = [random.randint(0, 255) for _ in range(3)] + [255]  # RGBA with A=255
        result[label] = {
            "id": str(uuid.uuid4()),
            "short_label_code": label,
            "long_label_code": label,
            "color": color
        }

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full file path
    file_path = os.path.join(output_dir, "class_mapping.json")

    # Write the JSON to the file
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Class mapping saved to {file_path}")


def copy_file(row, dataset, output_dir):
    """

    :param row:
    :param dataset:
    :param output_dir:
    :return:
    """
    src_path = row['Path']
    label = row['Label']
    img_name = os.path.basename(src_path)
    dst_path = f"{output_dir}/{dataset}/{label}/{img_name}"

    try:
        shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error copying image {src_path}: {e}")


def to_yolo(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("To YOLO")
    print("###############################################\n")
    # Create the output directories
    name = args.output_name if args.output_name else get_now()
    output_dir = f"{args.output_dir}/yolo/{name}"

    train_dir = f"{output_dir}/train"
    val_dir = f"{output_dir}/val"
    test_dir = f"{output_dir}/test"

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"NOTE: Found {len(args.patches)} patch files")

    # If the user provides multiple patch dataframes
    patches_df = pd.DataFrame()

    for patches_path in args.patches:
        if os.path.exists(patches_path):
            # Patch dataframe
            patches = pd.read_csv(patches_path)
            patches = patches.dropna()
            patches_df = pd.concat((patches_df, patches))
        else:
            print(f"WARNING: Patches dataframe {patches_path} does not exist")

    class_names = sorted(patches_df['Label'].unique())
    num_classes = len(class_names)

    # Create class mapping
    create_class_mapping(class_names, output_dir)

    # ------------------------------------------------------------------------------------------------------------------
    # Loading data, creating datasets
    # ------------------------------------------------------------------------------------------------------------------
    print("\n###############################################")
    print("Creating Datasets")
    print("###############################################\n")

    var = 'Image Name'

    # Names of all images; sets to be split based on images
    image_names = patches_df[var].unique()

    # Split the Images into training, validation, and test sets (70/20/10)
    # We split based on the image names, so that we don't have the same image in multiple sets.
    training_images, temp_images = train_test_split(image_names, test_size=0.1, random_state=42)
    validation_images, testing_images = train_test_split(temp_images, test_size=0.33, random_state=42)

    # Create training, validation, and test dataframes
    train_df = patches_df[patches_df[var].isin(training_images)]
    valid_df = patches_df[patches_df[var].isin(validation_images)]
    test_df = patches_df[patches_df[var].isin(testing_images)]

    if args.even_dist:
        # Downsample majority classes to make even distribution (+/- N%)
        train_df = downsample_majority_classes(train_df, about=args.about)
        valid_df = downsample_majority_classes(valid_df, about=args.about)
        test_df = downsample_majority_classes(test_df, about=args.about)

    train_classes = len(set(train_df['Label'].unique()))
    valid_classes = len(set(valid_df['Label'].unique()))
    test_classes = len(set(test_df['Label'].unique()))

    # If there isn't one class sample in each data sets
    # will throw an error; hacky way of fixing this.
    if not (train_classes == valid_classes == test_classes):
        print("NOTE: Sampling one of each class category")
        # Holds one sample of each class category
        sample = pd.DataFrame()
        # Gets one sample from patches_df
        for label in patches_df['Label'].unique():
            one_sample = patches_df[patches_df['Label'] == label].sample(n=1)
            sample = pd.concat((sample, one_sample))

        train_df = pd.concat((sample, train_df))
        valid_df = pd.concat((sample, valid_df))
        test_df = pd.concat((sample, test_df))

    # Reset the indices
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Output to logs
    train_df.to_csv(f"{output_dir}/Training_Set.csv", index=False)
    valid_df.to_csv(f"{output_dir}/Validation_Set.csv", index=False)
    test_df.to_csv(f"{output_dir}/Testing_Set.csv", index=False)

    # The number of class categories
    print(f"NOTE: Number of classes in training set is {len(train_df['Label'].unique())}, N={len(train_df)}")
    print(f"NOTE: Number of classes in validation set is {len(valid_df['Label'].unique())}, N={len(valid_df)}")
    print(f"NOTE: Number of classes in testing set is {len(test_df['Label'].unique())}, N={len(test_df)}")

    # ------------------------------------------------------------------------------------------------------------------
    # Data Exploration
    # ------------------------------------------------------------------------------------------------------------------
    plt.figure(figsize=(int(10 + num_classes * 0.5), 15))

    # Set the same y-axis limits for all subplots
    ymin = 0
    ymax = train_df['Label'].value_counts().max() + 10

    # Plotting the train data
    plt.subplot(1, 3, 1)
    plt.title(f"Train: {len(train_df)} Classes: {len(train_df['Label'].unique())}")
    ax = train_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the valid data
    plt.subplot(1, 3, 2)
    plt.title(f"Valid: {len(valid_df)} Classes: {len(valid_df['Label'].unique())}")
    ax = valid_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Plotting the test data
    plt.subplot(1, 3, 3)
    plt.title(f"Test: {len(test_df)} Classes: {len(test_df['Label'].unique())}")
    ax = test_df['Label'].value_counts().plot(kind='bar')
    ax.set_ylim([ymin, ymax])

    # Saving and displaying the figure
    plt.savefig(f"{output_dir}/DatasetSplit.jpg")
    plt.close()

    if os.path.exists(f"{output_dir}/DatasetSplit.jpg"):
        print(f"NOTE: Data split Figure saved in {output_dir}")

    # ------------------------------------------------------------------------------------------------------------------
    # Save patches
    # ------------------------------------------------------------------------------------------------------------------

    print("NOTE: Copying images to YOLO format...")

    # Create all necessary directories first
    for dataset in ["train", "val", "test"]:
        for label in class_names:
            os.makedirs(f"{output_dir}/{dataset}/{label}", exist_ok=True)

    # Use multiprocessing to copy files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for dataset, df in [("train", train_df), ("val", valid_df), ("test", test_df)]:
            copy_func = partial(copy_file, dataset=dataset, output_dir=output_dir)
            pool.map(copy_func, [row for _, row in df.iterrows()])

    # Create labels.txt file
    with open(f"{output_dir}/labels.txt", 'w') as f:
        for label in class_names:
            f.write(f"{label}\n")

    print(f"NOTE: YOLO formatted dataset created in {output_dir}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Create YOLO Dataset')

    parser.add_argument('--patches', required=True, nargs="+",
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument('--even_dist', action='store_true',
                        help='Downsample majority classes to be about +/- N% of minority class')

    parser.add_argument('--about', type=float, default=0.25,
                        help='Downsample majority classes by "about" +/- N% of minority class')

    parser.add_argument('--output_name', type=str, required=False,
                        help='Name to provide output directory (optional)')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save updated label csv file.')

    args = parser.parse_args()

    try:
        to_yolo(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
