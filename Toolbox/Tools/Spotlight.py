import os
import argparse
import warnings
import traceback
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

from renumics import spotlight as sp
from renumics.spotlight import layout, layouts
from renumics.spotlight.layout import lenses

import torch
import torchvision
import torch.nn.functional as F

import segmentation_models_pytorch as smp

from Classification import CustomModel
from Classification import get_classifier_encoders
from Classification import get_validation_augmentation

from Common import console_user

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def load_model(model_path):
    """
    User provides either pre-trained model (.pth), or an encoder name.

    :param model_path:
    :return:
    """
    if os.path.exists(model_path):
        # Loading the previously trained model (.pth)
        model = torch.load(model_path, map_location='cpu')

        encoder_name = model.name
        state_dict = model.encoder.state_dict()

        # Loading the state from provided model
        model.encoder.load_state_dict(state_dict, strict=True)
        print(f"NOTE: Loaded {model.name} with custom pre-trained weights")

    elif model_path in get_classifier_encoders():
        # Building model from imagenet encoder_name
        encoder_name = model_path
        model = CustomModel(encoder_name=model_path,
                            weights='imagenet',
                            dropout_rate=0,
                            class_names=[""])

        print(f"NOTE: Loaded {model.name} with imagenet pre-trained weights")

    else:
        raise Exception("ERROR: Provide pre-trained model path or encoder name")

    # Convert patches to PyTorch tensor with validation augmentation and preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 'imagenet')

    return model, preprocessing_fn


def calc_bboxes(df):
    """
    Creates bounding boxes for the image using patch location and size.

    :param df:
    :return:
    """
    bboxes = []

    for i, r in df.iterrows():
        patch_w, patch_h = Image.open(r['Path']).size
        image_w, image_h = Image.open(r['Image Path']).size

        xmin = (r['Column'] - patch_w / 2) / image_w
        ymin = (r['Row'] - patch_h / 2) / image_h
        xmax = (r['Column'] + patch_w / 2) / image_w
        ymax = (r['Row'] + patch_h / 2) / image_h

        bbox = [xmin, ymin, xmax, ymax]
        bboxes.append(bbox)

    df['Box'] = bboxes

    return df


def load_patches(paths):
    """
    Loads a batch of patches as numpy arrays.

    :param paths:
    :return:
    """
    patches = []

    for path in paths:
        # Check that the path exists, otherwise ignore
        if not os.path.exists(path):
            print(f"WARNING: {path} does not exist, skipping.")
            continue

        # Read patch and add to list
        patches.append(np.array(Image.open(path)))

    if not patches:
        raise Exception(f"ERROR: Patch files not found, check provided csv")

    return patches


def get_feature_embeddings(patches_df, model, preprocessing_fn, validation_augmentation, device='cuda'):
    """
    Extract feature embeddings for image patches using the provided model.

    :param patches_df: DataFrame containing patch information (Name, Path, Label)
    :param model: The model used for feature extraction
    :param preprocessing_fn: Function to preprocess the images
    :param validation_augmentation: Augmentation function for validation
    :param device: Device to run the model on (default: 'cuda')
    :return: DataFrame with added feature embeddings
    """
    print(f"NOTE: Creating feature embeddings using {model.name}")

    model.eval()
    model.to(device)

    # Create a copy of the input DataFrame
    df = patches_df.copy()

    # Function to process a batch of patches
    def process_batch(batch_paths):
        patches = load_patches(batch_paths)
        patches_tensor = [validation_augmentation(image=p)['image'] for p in patches]
        patches_tensor = [preprocessing_fn(p) for p in patches_tensor]
        patches_tensor = torch.stack([torch.Tensor(p) for p in patches_tensor]).permute(0, 3, 1, 2)
        patches_tensor = patches_tensor.to(device)

        with torch.no_grad():
            features = model.encoder(patches_tensor)[-1]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            return features.cpu().numpy()

    # Process patches in batches
    batch_size = 256
    feature_list = []

    for i in tqdm(range(0, len(df), batch_size), desc="Extracting features"):
        batch_paths = df['Path'].iloc[i:i+batch_size].tolist()
        batch_features = process_batch(batch_paths)
        feature_list.extend(batch_features)

    torch.cuda.empty_cache()

    return feature_list


def spotlight(args):
    """

    :param args:
    :return:
    """
    print("\n###############################################")
    print("Spotlight")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the patches
    if os.path.exists(args.patches):

        # Ensure frac is [0-1]
        frac = args.frac if type(args.frac) == float else args.frac / 100

        # Patch dataframe
        patches_df = pd.read_csv(args.patches, index_col=0).sample(frac=frac)
        patches_df = calc_bboxes(patches_df)

        # Ensure quality of patches.csv
        assert "Path" in patches_df.columns, print(f"ERROR: 'Path' not in provided csv")
        assert "Name" in patches_df.columns, print(f"ERROR: 'Image Name' not in provided csv")
        assert "Column" in patches_df.columns, print(f"ERROR: 'Column' not in provided csv")
        assert "Row" in patches_df.columns, print(f"ERROR: 'Row' not in provided csv")
    else:
        raise Exception(f"ERROR: Patches dataframe {args.patches} does not exist")

    # Model Weights
    if args.pre_trained_path:
        model_path = args.pre_trained_path
    elif args.encoder_name:
        model_path = args.encoder_name
    else:
        raise Exception("ERROR: Provide pre-trained model path (.pth) or encoder name")

    model, preprocessing_fn = load_model(model_path)
    validation_augmentation = get_validation_augmentation(height=224, width=224)

    # Get the feature embeddings for all patches in the data frame
    patches_df['Embeddings'] = get_feature_embeddings(patches_df, model, preprocessing_fn, validation_augmentation)

    # Setup spotlight
    patch_lense = lenses.image("Path")
    image_lense = lenses.bounding_box("Image Path", "Box")
    lense_list = [patch_lense, image_lense]

    default_layout = layouts.default()
    default_layout.children[1].children[0] = layout.inspector(lenses=lense_list)

    # Display
    sp.show(patches_df, embed=['Embeddings'], layout=default_layout)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """
    parser = argparse.ArgumentParser(description="Spotlight")

    parser.add_argument("--pre_trained_path", type=str, default="",
                        help="Path to existing Best Model and Weights File (.pth)")

    parser.add_argument('--encoder_name', type=str, default='efficientnet-b0',
                        help='The convolutional encoder to use; pretrained on Imagenet')

    parser.add_argument('--patches', type=str,
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument('--frac', type=float, default=0.1,
                        help='The fraction of patches to use in spotlight')

    args = parser.parse_args()

    try:

        spotlight(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()