import os
import argparse
import ujson as json
from tqdm import tqdm
from collections import Counter

import math


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def count_images_per_class(src_dir):
    """
    Count the number of images per class in a dataset.
    """
    class_counts = Counter()
    for root, dirs, files in tqdm(os.walk(src_dir), desc="Walking through directories"):
        if not dirs:  # We are in a class directory
            class_name = os.path.basename(root)
            class_counts[class_name] += len(files)
    return class_counts


def calculate_gini_index(class_counts):
    """
    Calculate the GINI index for a set of class counts.
    """
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return 0.0

    # Filter out classes with 0 samples
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    if not non_zero_counts:
        return 0.0

    sum_squared_probs = sum((count / total_samples) ** 2 for count in non_zero_counts)
    gini_index = 1 - sum_squared_probs
    return gini_index


def calculate_shannon_index(class_counts):
    """
    Calculate the Shannon index for a set of class counts.
    """
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return 0.0

    # Filter out classes with 0 samples
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    if not non_zero_counts:
        return 0.0

    shannon_index = 0
    for count in non_zero_counts:
        p = count / total_samples
        shannon_index += -p * math.log(p)
    return shannon_index


def calculate_normalized_shannon_index(class_counts):
    """
    Calculate the normalized Shannon index for a set of class counts.
    """
    total_samples = sum(class_counts.values())
    if total_samples == 0:
        return 0.0

    # Filter out classes with 0 samples
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    if not non_zero_counts:
        return 0.0

    shannon_index = calculate_shannon_index(class_counts)
    max_shannon_index = math.log(len(non_zero_counts))
    normalized_shannon_index = shannon_index / max_shannon_index
    return normalized_shannon_index


def main():
    parser = argparse.ArgumentParser(description="Calculate GINI index for a dataset.")
    parser.add_argument("--src", type=str, help="Source directory of the dataset.")
    args = parser.parse_args()

    src_dir = args.src
    dataset_types = ['train', 'val', 'test']
    results = {}
    combined_class_counts = Counter()

    for dataset_type in tqdm(dataset_types, desc="Processing dataset types"):
        dataset_path = os.path.join(src_dir, dataset_type)
        class_counts = count_images_per_class(dataset_path)
        gini_index = calculate_gini_index(class_counts)
        shannon_index = calculate_shannon_index(class_counts)
        normalized_shannon_index = calculate_normalized_shannon_index(class_counts)
        results[dataset_type] = {
            'class_counts': dict(class_counts),
            'total_samples': sum(class_counts.values()),
            'gini_index': gini_index,
            'shannon_index': shannon_index,
            'normalized_shannon_index': normalized_shannon_index,
        }
        combined_class_counts.update(class_counts)

    combined_gini_index = calculate_gini_index(combined_class_counts)
    combined_shannon_index = calculate_shannon_index(combined_class_counts)
    combined_normalized_shannon_index = calculate_normalized_shannon_index(combined_class_counts)
    results['combined'] = {
        'class_counts': dict(combined_class_counts),
        'total_samples': sum(combined_class_counts.values()),
        'gini_index': combined_gini_index,
        'shannon_index': combined_shannon_index,
        'normalized_shannon_index': combined_normalized_shannon_index
    }

    output_path = os.path.join(src_dir, "entropy_index_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
