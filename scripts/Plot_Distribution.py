import os
import argparse
import ujson as json

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def plot_distributions(json_path):
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.ravel()

    # Plot each dataset type
    for idx, dataset_type in enumerate(['train', 'val', 'test', 'combined']):
        counts = data[dataset_type]['class_counts']

        # Sort by count descending
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        classes, values = zip(*sorted_items)

        # Create bar plot
        axes[idx].bar(range(len(values)), values)
        axes[idx].set_xticks(range(len(classes)))
        axes[idx].set_xticklabels(classes, rotation=45, ha='right')
        axes[idx].set_title(f'{dataset_type.capitalize()} Distribution')
        axes[idx].set_ylabel('Number of Images')

        # Add metrics text
        metrics_text = f"Norm Shannon: {data[dataset_type]['normalized_shannon_index']:.3f}"
        axes[idx].text(0.95, 0.95, metrics_text,
                       transform=axes[idx].transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot in the same directory as the JSON file
    output_path = os.path.join(os.path.dirname(json_path), 'class_distributions.png')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot class distributions from a JSON file.')
    parser.add_argument('--src', required=True, help='Path to the JSON file')
    args = parser.parse_args()

    plot_distributions(args.src)
