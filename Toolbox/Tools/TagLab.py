import os
import sys
import warnings
import argparse
import traceback

import numpy as np

import cv2
from skimage.io import imread
from skimage.io import imsave
from PIL import Image

import torch
import torchvision

from MSS import get_sam_predictor

from Common import log

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def taglab(args):
    """

    """
    log("\n###############################################")
    log("TagLab w/ SAM")
    log("###############################################\n")

    # Check for CUDA
    log(f"NOTE: PyTorch version - {torch.__version__}")
    log(f"NOTE: Torchvision version - {torchvision.__version__}")
    log(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Image file
    if os.path.exists(args.orthomosaic):
        input_path = args.orthomosaic
        orthomosaic = imread(input_path, plugin='pil')
        height, width = orthomosaic.shape[0:2]
    else:
        log("ERROR: Orthomosaic path provided doesn't exist.")
        sys.exit(1)

    # Model Weights
    try:
        # Load the model with custom metrics
        sam_predictor = get_sam_predictor(args.model_type,
                                          device,
                                          points_per_side=args.points_per_side,
                                          points_per_batch=args.points_per_batch)

        log(f"NOTE: Loaded model {args.model_type}")

    except Exception as e:
        log(f"ERROR: There was an issue loading the model\n{e}")
        sys.exit(1)

    # Setting output directories
    output_dir = f"{args.output_dir}\\taglab\\"
    os.makedirs(output_dir, exist_ok=True)

    # Output file
    output_basename = os.path.basename(input_path).split('.')[0]
    output_path = f"{output_dir}{output_basename}.png"

    # ----------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------
    log("\n###############################################")
    log("Making Mask")
    log("###############################################\n")

    # Resize the original
    orthomosaic = cv2.resize(orthomosaic, (1024, 1024))

    # Mask to store the results
    output_mask = np.full(shape=(1024, 1024, 4), fill_value=255, dtype=np.uint8)

    # TagLab expected blank RGBAs
    output_mask[:, :, 0] = 0
    output_mask[:, :, 1] = 15
    output_mask[:, :, 2] = 15

    # Use sam to generate masks
    masks = sam_predictor.generate(orthomosaic)

    # Loop though, add to output mask
    for mask in masks:
        segment = mask['segmentation']
        output_mask[segment, 0:3] = [54, 54, 54]  # Other class

    # Resize using nn, then save
    output_mask = cv2.resize(output_mask, (height, width), interpolation=cv2.INTER_NEAREST)
    imsave(fname=output_path, arr=output_mask.astype(np.uint8))
    log(f"NOTE: Saved seg mask to {output_path}")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="TagLab w/ SAM")

    parser.add_argument("--orthomosaic", type=str, required=True,
                        help="Directory containing images to perform inference on")

    parser.add_argument("--model_type", type=str, default='vit_b',
                        help="Model to use; one of ['vit_b', 'vit_l', 'vit_h']")

    parser.add_argument("--points_per_side", type=int, default=64,
                        help="The number of points to sample from image (power of two)")

    parser.add_argument("--points_per_batch", type=int, default=64,
                        help="The number of points per batch (power of two)")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where predictions will be saved.")

    args = parser.parse_args()

    try:
        taglab(args)
        log("Done.\n")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()
