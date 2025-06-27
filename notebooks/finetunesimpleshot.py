# -*- coding: utf-8 -*-
"""
This script fine-tunes a SimpleShot model using a local, YOLO-formatted 
image classification dataset.

It combines the original FineTuneSimpleShot notebook with a custom data loader.
"""

import os
import sys
import itertools
from tqdm import tqdm  # Use standard tqdm for scripts

import numpy as np
import matplotlib.pyplot as plt

import sklearn.neighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import torch
from datasets import load_dataset, DatasetDict
from bioclip.predict import CustomLabelsClassifier
from bioclip.predict import BaseClassifier


# ==============================================================================
# UTILITY FUNCTION TO LOAD YOLO-FORMATTED DATASET
# ==============================================================================


def load_yolo_classification_dataset(dataset_dir: str) -> DatasetDict:
    """
    Loads a YOLO-formatted image classification dataset from a local directory.

    The expected directory structure is:
    dataset_dir/
    ├── train/
    │   ├── class_a/
    │   │   └── *.jpg
    │   └── class_b/
    │       └── *.jpg
    ├── valid/  (or val, or test)
    │   ├── class_a/
    │   │   └── *.jpg
    │   └── class_b/
    │       └── *.jpg

    This function standardizes the split names to 'train' and 'test' to match
    the fine-tuning script's expectations. It prioritizes 'test' if it exists,
    otherwise it renames 'validation' or 'val' to 'test'.

    Args:
        dataset_dir (str): The path to the root directory of the dataset.

    Returns:
        datasets.DatasetDict: A dataset object ready for use.
    """
    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    print(f"Loading dataset from local directory: {dataset_dir}")
    # Load the dataset using the 'imagefolder' builder which infers classes
    dataset = load_dataset("imagefolder", data_dir=dataset_dir, drop_labels=False)

    # Standardize the split names for compatibility with the script
    if 'train' not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")

    # Find and standardize the test/validation split
    test_split_key = None
    if 'test' in dataset:
        test_split_key = 'test'
    elif 'validation' in dataset:
        test_split_key = 'validation'
    elif 'val' in dataset:
        test_split_key = 'val'

    if not test_split_key:
        print("Warning: No 'test', 'validation', or 'val' split found. Using 'train' split for testing as a fallback.")
        dataset['test'] = dataset['train']
    elif test_split_key != 'test':
        print(f"Renaming split '{test_split_key}' to 'test' for consistency.")
        dataset['test'] = dataset.pop(test_split_key)

    # Create a new DatasetDict with only the required splits
    final_dataset = DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })

    print("Dataset loaded and splits standardized:")
    print(final_dataset)

    return final_dataset


# ==============================================================================
# CONFIGURATION
# ==============================================================================


# --- USER CONFIGURATION ---
# Set this to the path of your YOLO dataset directory.
# If you leave it as None, the script will run a self-contained demo.
YOLO_DATASET_DIR = "C:\\Users\\jordan.pierce\\Downloads\\DelValle_Polyps\\YOLO_Classification_Dataset"

# --- SCRIPT SETTINGS ---
TRAIN_NAME = 'train'  # Name of the training split
TEST_NAME = 'test'    # Name of the test split
IMAGE_NAME = 'image'  # Key for image data
LABEL_NAME = 'label'  # Key for label data
BATCH_SIZE = 30       # Batch size for feature extraction


# ==============================================================================
# MAIN FINE-TUNING SCRIPT
# ==============================================================================


# --- Determine GPU or CPU ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

try:
    dataset = load_yolo_classification_dataset(YOLO_DATASET_DIR)
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Please make sure the `YOLO_DATASET_DIR` variable is set correctly.")
    sys.exit(1)

print("\nLabels:", ",".join(dataset[TRAIN_NAME].features[LABEL_NAME].names))
print("Example Image:", dataset[TRAIN_NAME][IMAGE_NAME][0])


# --- Function to Convert PIL to image embeddings ---
classifier = BaseClassifier(device=device)


def batched(items, batch_size):
    """Yield successive batches from a list of items."""
    # This function splits a list into smaller batches of a given size
    it = iter(items)
    while (batch := list(itertools.islice(it, batch_size))):
        yield batch


def create_image_features(pil_image_ary):
    """Create image embeddings for a list of PIL images."""
    # This function takes a list of images and returns their feature vectors (embeddings)
    # List to collect all feature tensors from batches
    all_features = []
    # Use tqdm to show progress bar for embedding creation
    with tqdm(total=len(pil_image_ary), desc='Creating image embeddings', unit='image') as progress_bar:
        # Process images in batches for efficiency
        for images in batched(pil_image_ary, BATCH_SIZE):
            with torch.no_grad():
                # Extract features for the current batch, with normalization
                features = classifier.create_image_features(images, normalize=True)
            # Move features to CPU and add to the list
            all_features.append(features.cpu())
            # Update progress bar
            progress_bar.update(len(images))
    # Concatenate all batch features into a single numpy array and return
    return torch.cat(all_features, dim=0).cpu().numpy()


def show_metrics(expected_label_ary, predicted_label_ary, title="Confusion Matrix"):
    """Display accuracy and confusion matrix for predictions."""
    # Print the overall accuracy score
    print("Accuracy:", round(accuracy_score(expected_label_ary, predicted_label_ary), 3))
    
    # Compute the confusion matrix
    cm = confusion_matrix(expected_label_ary, predicted_label_ary)
    
    # Create a display object for the confusion matrix with class labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset[TRAIN_NAME].features[LABEL_NAME].names)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation='vertical')
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.show()


def l2_normalize(features):
    """L2-normalize a numpy array of features."""
    # This function scales each feature vector to have length 1
    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


class SimpleShot(object):
    """SimpleShot nearest-centroid classifier for image features."""
    def __init__(self, device):
        # Initialize the classifier and set the device (CPU/GPU)
        self.device = device
        self.x_mean = None
        self.centroids = None

    def mean_normalize(self, x):
        """Subtract mean and L2-normalize features."""
        # Center the features by subtracting the mean, then normalize
        return l2_normalize(x - self.x_mean)

    def fit(self, pil_images, labels):
        """Fit centroids to training images and labels."""
        print("Extracting features from training images...")
        # Extract features from training images
        x_train = create_image_features(pil_images)
        
        print("Computing mean and normalizing features...")
        # Compute the mean feature vector
        self.x_mean = x_train.mean(axis=0, keepdims=True)
        # Normalize the features
        x_norm = self.mean_normalize(x_train)

        print("Fitting nearest centroid classifier...")
        # Use NearestCentroid classifier to find class centroids
        clf = sklearn.neighbors.NearestCentroid()
        clf.fit(x_norm, labels)
        # Store the centroids as a tensor on the correct device
        self.centroids = torch.from_numpy(
            clf.centroids_).type(torch.float32).to(self.device)
        print("Training complete!")

    def predict(self, pil_images):
        """Predict labels for a list of PIL images."""
        # Extract features from test images
        if self.x_mean is None:
            raise ValueError("You must call fit() before running predict().")
        
        print("Extracting features from test images...")
        x_test = create_image_features(pil_images)
        
        print("Normalizing test features...")
        # Normalize the features
        x_norm = self.mean_normalize(x_test)
        x_test_tensor = torch.from_numpy(x_norm).to(self.device)

        print("Computing distances to centroids...")
        # Compute distances to each centroid and pick the closest
        distances = torch.linalg.vector_norm(x_test_tensor[:, None] - self.centroids, axis=2)
        preds = torch.argmin(distances, dim=1)
        print("Prediction complete!")
        return preds.cpu().numpy()


# --- Train the model ---
print("\n--- Training the SimpleShot model ---")
simpleshot = SimpleShot(device)
simpleshot.fit(dataset[TRAIN_NAME][IMAGE_NAME], dataset[TRAIN_NAME][LABEL_NAME])

# --- Create predictions ---
print("\n--- Evaluating the Fine-Tuned SimpleShot Model ---")
predicted_labels_simpleshot = simpleshot.predict(dataset[TEST_NAME][IMAGE_NAME])
show_metrics(dataset[TEST_NAME][LABEL_NAME], predicted_labels_simpleshot, "Confusion Matrix (Fine-Tuned SimpleShot)")

# --- Compare against untrained pybioclip model ---
print("\n--- Evaluating the Untrained BioCLIP Model for Comparison ---")
untrained_classifier = CustomLabelsClassifier(dataset[TEST_NAME].features[LABEL_NAME].names, device=device)

# Predict labels for the test set using the untrained BioCLIP model
predicted_labels_untrained = []
with tqdm(total=len(dataset[TEST_NAME][IMAGE_NAME]), desc='Predicting with BioCLIP', unit='image') as progress_bar:
    for images in batched(dataset[TEST_NAME][IMAGE_NAME], BATCH_SIZE):
        with torch.no_grad():
            # Get predictions from the untrained classifier (top-1 prediction)
            predictions = untrained_classifier.predict(images, k=1)
        for pred in predictions:
            # Convert predicted class name to integer label
            label_str = pred['classification']
            label = dataset[TEST_NAME].features[LABEL_NAME].str2int(label_str)
            predicted_labels_untrained.append(label)
        progress_bar.update(len(images))

show_metrics(dataset[TEST_NAME][LABEL_NAME], predicted_labels_untrained, "Confusion Matrix (Untrained BioCLIP)")