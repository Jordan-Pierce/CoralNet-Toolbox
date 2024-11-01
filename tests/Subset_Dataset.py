import os
import shutil
import argparse
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from tqdm import tqdm


def count_images_per_class(src_dir):
    class_counts = {}
    for root, dirs, files in os.walk(src_dir):
        if not dirs:  # We are in a class directory
            class_name = os.path.basename(root)
            class_counts[class_name] = len(files)
    return class_counts


def subset_dataset(src_dir, dst_dir, min_count):
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            class_dir = os.path.join(root, dir_name)
            subset_class_dir = class_dir.replace(src_dir, dst_dir)
            os.makedirs(subset_class_dir, exist_ok=True)

        for file_name in tqdm(files, desc=f"Processing {os.path.basename(root)}"):
            class_name = os.path.basename(root)
            class_dir = os.path.join(dst_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            if class_name in min_count and len(os.listdir(class_dir)) < min_count[class_name]:
                src_file = os.path.join(root, file_name)
                dst_file = os.path.join(class_dir, file_name)
                shutil.copy(src_file, dst_file)


def plot_distribution(dst_dir, dataset_type):
    class_counts = count_images_per_class(os.path.join(dst_dir, dataset_type))
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])

    plt.figure(figsize=(10, 6))
    plt.bar(df['Class'], df['Count'], color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(f'Distribution of {dataset_type.capitalize()} Set')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(dst_dir, f'{dataset_type}_distribution.png')
    plt.savefig(plot_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Subset a YOLO formatted dataset.")
    parser.add_argument("src", type=str, help="Source directory of the YOLO formatted dataset.")
    parser.add_argument("dst", type=str, help="Destination directory for the subsetted dataset.")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = os.path.join(os.path.dirname(src_dir), "Datasets_Subsetted", args.dst)

    # Count images per class
    class_counts = count_images_per_class(src_dir)

    # Find the minority class count greater than 1
    min_count = min([count for count in class_counts.values() if count > 1])

    # Create a dictionary with the minimum count for each class
    min_count_dict = {class_name: min_count for class_name in class_counts}

    # Print the amount per class
    print("Original Class Counts:")
    for class_name, count in class_counts.items():
        print(f"Class: {class_name}, Images: {count}")

    print("\nSubsetted Class Counts:")
    for class_name, count in min_count_dict.items():
        print(f"Class: {class_name}, Images: {count}")

    # Subset the dataset
    subset_dataset(src_dir, dst_dir, min_count_dict)

    # Plot and save distribution for train, valid, and test sets
    for dataset_type in ['train', 'valid', 'test']:
        plot_distribution(dst_dir, dataset_type)

    print(f"\nSubsetted dataset created at {dst_dir}")


if __name__ == "__main__":
    main()