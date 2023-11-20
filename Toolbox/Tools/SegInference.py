import os
import sys
import json
import glob
import warnings
import argparse
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io

import torch
import torchvision
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp

from Common import log
from Common import get_now
from Common import print_progress
from Common import IMG_FORMATS

from Segmentation import colorize_mask
from Segmentation import get_preprocessing
from Segmentation import get_validation_augmentation

torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ------------------------------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------------------------------

class Dataset(BaseDataset):
    """

    """

    def __init__(
            self,
            dataframe,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        assert 'Name' in dataframe.columns, print(f"ERROR: 'Name' not found in mask file")
        assert 'Image Path' in dataframe.columns, print(f"ERROR: 'Semantic Path' not found in mask file")

        self.ids = dataframe['Name'].to_list()
        self.images_paths = dataframe['Image Path'].to_list()

        # convert str names to class values on masks
        self.class_ids = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return torch.from_numpy(image)

    def __len__(self):
        return len(self.ids)


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def segmentation_inference(args):
    """

    """
    print("\n###############################################")
    print("Semantic Segmentation")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Image files
    if os.path.exists(args.images):
        # Directory containing images
        image_dir = args.images
        # Get images from directory
        image_paths = [i for i in glob.glob(f"{image_dir}/*.*") if i.split(".")[-1].lower() in IMG_FORMATS]

        if not image_paths:
            raise Exception(f"ERROR: No images were found in the directory provided; please check input.")
        else:
            print(f"NOTE: Found {len(image_paths)} images in directory provided")

        # Get the names of the images
        image_names = [os.path.basename(i) for i in image_paths]

        # Create a dataframe
        test_df = pd.DataFrame(list(zip(image_names, image_paths)), columns=['Name', 'Image Path'])
    else:
        print("ERROR: Directory provided doesn't exist.")
        sys.exit(1)

    # Color mapping file
    if os.path.exists(args.color_map):
        with open(args.color_map, 'r') as json_file:
            color_map = json.load(json_file)

        # Modify color map format
        class_names = list(color_map.keys())
        class_ids = [color_map[c]['id'] for c in class_names]
        class_colors = [color_map[c]['color'] for c in class_names]

    else:
        print(f"ERROR: Color Mapping JSON file provided doesn't exist; check input provided")
        sys.exit(1)

    # Model weights, load it up
    if os.path.exists(args.model):

        # Load into the model
        model = torch.load(args.model)
        model_name = "-".join(model.name.split("-")[1:])
        print(f"NOTE: Loaded weights {model.name}")

        # Get the preprocessing function that was used during training
        preprocessing_fn = smp.encoders.get_preprocessing_fn(model_name, 'imagenet')

    else:
        raise Exception(f"ERROR: Model path provided does not exists; check input provided")

    # Setting output directories
    output_dir = f"{args.output_dir}\\masks\\Segmentation_{get_now()}\\"
    seg_dir = f"{output_dir}\\semantic\\"
    mask_dir = f"{output_dir}\\mask\\"
    color_dir = f"{output_dir}\\color\\"
    overlay_dir = f"{output_dir}\\overlay\\"

    # Output dataframe
    output_mask_csv = f"{output_dir}masks.csv"

    # Create the output directories
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Sample augmentation techniques
    height, width = 736, 1280

    # Open an image using PIL to get the original dimensions
    original_width, original_height = Image.open(test_df.loc[0, 'Image Path']).size

    # Create test dataset
    test_dataset = Dataset(
        test_df,
        augmentation=get_validation_augmentation(height, width),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=class_ids,
    )

    # Output dataframe
    mask_df = []

    for _ in range(len(test_dataset)):

        # Image name
        image_name = test_dataset.ids[_]
        # Image path
        image_path = test_dataset.images_paths[_]
        # Original image
        image_og = io.imread(image_path)

        # Augmented image
        image = test_dataset[_]
        # Prepare sample
        x_tensor = image.to(device).unsqueeze(0)

        # Make prediction
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = np.argmax(pr_mask, axis=0)

        # Resize
        pr_mask = cv2.resize(pr_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # ------------------------------------------------
        # Save the final masks
        # ------------------------------------------------

        # Save the semantic mask
        seg_mask = pr_mask.astype(np.uint8)
        semantic_path = f"{seg_dir}{image_name.split('.')[0]}.png"
        io.imsave(fname=semantic_path, arr=seg_mask)
        print(f"NOTE: Saved semantic mask to {semantic_path}")

        # Save the traditional mask (0 background, 255 object)
        mask = np.zeros(shape=(original_height, original_width, 3), dtype=np.uint8)
        mask[pr_mask != 0] = [255, 255, 255]
        mask_path = f"{mask_dir}{image_name.split('.')[0]}.png"
        io.imsave(fname=mask_path, arr=mask.astype(bool))
        print(f"NOTE: Saved mask to {mask_path}")

        # Save the color mask
        color_mask = colorize_mask(pr_mask, class_ids, class_colors)
        color_mask[pr_mask == 0, :] = [0, 0, 0]
        color_path = f"{color_dir}{image_name.split('.')[0]}.png"
        io.imsave(fname=color_path, arr=color_mask.astype(np.uint8))
        print(f"NOTE: Saved color mask to {color_path}")

        # Save the overlay mask
        overlay_mask = cv2.addWeighted(image_og, 0.5, color_mask, 0.5, 0)
        overlay_path = f"{overlay_dir}{image_name.split('.')[0]}.png"
        io.imsave(fname=overlay_path, arr=overlay_mask.astype(np.uint8))
        print(f"NOTE: Saved overlay to {overlay_path}")

        mask_df.append([image_name, image_path, semantic_path, mask_path, color_path, overlay_path])

        # Gooey
        print_progress(_, len(test_dataset))

    # Save dataframe to root directory
    mask_df = pd.DataFrame(mask_df, columns=['Name', 'Image Path', 'Semantic Path',
                                             'Mask Path', 'Color Path', 'Overlay Path'])
    mask_df.to_csv(output_mask_csv)

    if os.path.exists(output_mask_csv):
        print(f"NOTE: Mask dataframe saved to {output_dir}")
    else:
        print(f"ERROR: Could not save mask dataframe")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Model Inference")

    parser.add_argument("--images", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to Best Model and Weights File (.pth)")

    parser.add_argument("--color_map", type=str, required=True,
                        help="Path to the model's Color Map JSON file")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        segmentation_inference(args)
        print("Done.\n")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()