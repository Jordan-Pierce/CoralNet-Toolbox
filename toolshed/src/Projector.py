import os
import argparse
import warnings
import traceback

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread

from tensorboard import program
from tensorboard.plugins import projector as P

import torch
import torchvision
import torch.nn.functional as F

import segmentation_models_pytorch as smp

from src.Common import get_now
from src.Common import console_user
from src.Common import progress_printer

from src.Classification import get_validation_augmentation

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------------------------------

class CustomModel(torch.nn.Module):
    def __init__(self, encoder_name, weights):
        super(CustomModel, self).__init__()

        # Name
        self.name = encoder_name

        # Pre-trained encoder
        self.encoder = smp.encoders.get_encoder(name=encoder_name,
                                                weights=weights)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(self.encoder.out_channels[-1], 100)

    # Add a method to get the name attribute
    def get_name(self):
        return self.name

    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)
        x = x[-1]
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # Fully connected layer for classification
        x = self.fc(x)
        # Softmax activation
        x = F.softmax(x, dim=1)


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def load_patches(paths):
    """

    """
    patches = []

    for path in paths:

        if not os.path.exists(path):
            print(f"WARNING: {path} does not exist, skipping.")

        # Read patch and add to list
        patches.append(imread(path))

    if not patches:
        raise Exception(f"ERROR: Patch files not found, check provided csv")

    return patches


def create_sprite_image(patches, output_path):
    """

    """
    #
    grid = int(np.sqrt(len(patches))) + 1
    sprite_height = 8192 // grid
    sprite_width = 8192 // grid

    big_image = Image.new(
        mode='RGB',
        size=(sprite_width * grid, sprite_height * grid),
        color=(0, 0, 0))

    print("NOTE: Creating sprites")

    for i in range(len(patches)):
        row = i // grid
        col = i % grid
        img = patches[i]
        img = img.resize((sprite_width, sprite_height))
        row_loc = row * sprite_height
        col_loc = col * sprite_width
        big_image.paste(img, (col_loc, row_loc))

    big_image.save(output_path, transparency=0)

    return sprite_width, sprite_height


def write_embeddings(logs_dir, patches, labels, features):
    """

    """
    print("NOTE: Setting up projector")

    metadata_file = f"{logs_dir}/metadata.csv"
    tensor_file = f"{logs_dir}/features.tsv"
    sprite_file = f"{logs_dir}/sprite.jpg"

    # Write the metadata
    with open(os.path.join(logs_dir, metadata_file), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    # Write the configs
    with open(os.path.join(logs_dir, tensor_file), "w") as f:
        for _, tensor in progress_printer(enumerate(features)):
            f.write("{}\n".format("\t".join(str(x) for x in tensor)))

    # Convert list of np arrays to list of Images
    pil_patches = [Image.fromarray(p) for p in patches]

    # Create projector config object
    config = P.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = metadata_file
    embedding.tensor_path = tensor_file
    sprite_width, sprite_height = create_sprite_image(pil_patches, sprite_file)
    embedding.sprite.image_path = os.path.basename(sprite_file)
    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([sprite_width, sprite_height])

    # Project the features using tensorboard
    P.visualize_embeddings(logs_dir, config)


def activate_tensorboard(logs_dir):
    """

    """
    # Activate tensorboard
    print("NOTE: Activating Tensorboard...")
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logs_dir])
    url = tb.launch()
    print(f"NOTE: View TensorBoard 2.10.1 at {url}")

    try:
        while True:
            continue

    except Exception:
        print("NOTE: Deactivating Tensorboard...")


def projector(args):
    """

    """

    print("\n###############################################")
    print("Projector")
    print("###############################################\n")

    # Check for CUDA
    print(f"NOTE: PyTorch version - {torch.__version__}")
    print(f"NOTE: Torchvision version - {torchvision.__version__}")
    print(f"NOTE: CUDA is available - {torch.cuda.is_available()}")

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------------------------------------------------------------------------------
    # If there's an existing project
    if args.project_folder:
        try:
            if os.path.exists(args.project_folder):
                print("NOTE: Opening existing project")
                activate_tensorboard(args.project_folder)
            else:
                raise Exception

        except Exception as e:
            raise Exception("ERROR: Project file provided does not exist, check provided input")

    # Source directory setup
    logs_dir = f"{args.output_dir}/projector/{get_now()}"
    os.makedirs(logs_dir)

    # Get the patches
    if os.path.exists(args.patches):
        # Patch dataframe
        patches_df = pd.read_csv(args.patches, index_col=0)
        patches_df = patches_df.dropna()
        # Get the image base names
        assert "Path" in patches_df.columns, print(f"ERROR: 'Path' not in provided csv")
        assert "Name" in patches_df.columns, print("ERROR: 'Image Name' not in provided csv")
        image_names = patches_df['Image Name'].unique().tolist()
    else:
        raise Exception(f"ERROR: Patches dataframe {args.patches} does not exist")

    # Model Weights
    if os.path.exists(args.model):

        try:
            # Loading the actual model / weights
            pre_trained_model = torch.load(args.model, map_location='cpu')

            try:
                # Getting the encoder name (preprocessing), and encoder state
                encoder_name = pre_trained_model.name
                state_dict = pre_trained_model.encoder.state_dict()
            except:
                encoder_name = pre_trained_model.encoder.name
                state_dict = pre_trained_model.encoder.encoder.state_dict()

            # Building model for projector
            encoder_weights = 'imagenet'
            model = CustomModel(encoder_name=encoder_name,
                                weights=encoder_weights)

            # Loading the state from provided model
            model.encoder.load_state_dict(state_dict, strict=True)
            print(f"NOTE: Loaded pre-trained weights from {encoder_name}")

            # Convert patches to PyTorch tensor with validation augmentation and preprocessing
            validation_augmentation = get_validation_augmentation(height=224, width=224)
            preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

            model.to(device)

        except Exception as e:
            raise Exception(f"ERROR: Could not build model\n{e}")
    else:
        raise Exception("ERROR: Model provided doesn't exist.")

    # Loop through each of the images, extract features from associated patches
    patches_sorted = []
    labels_sorted = []
    features_sorted = []

    print("NOTE: Creating feature embeddings")

    for i_idx, image_name in progress_printer(enumerate(image_names)):
        # Patches for current image
        current_patches = patches_df[patches_df['Image Name'] == image_name]
        # Get the patch labels
        labels = current_patches['Label'].values.tolist()
        # Get the patch arrays
        patches = np.stack(load_patches(current_patches['Path'].values.tolist()))

        # Augment
        patches_tensor = [validation_augmentation(image=p)['image'] for p in patches]
        # Pre-process
        patches_tensor = [preprocessing_fn(p) for p in patches_tensor]

        # Convert to tensors
        patches_tensor = [torch.Tensor(p) for p in patches_tensor]
        patches_tensor = torch.stack(patches_tensor).permute(0, 3, 1, 2)
        patches_tensor = patches_tensor.to(device)

        # Extract features from the encoder, reshape
        with torch.no_grad():
            features = model.encoder(patches_tensor)[-1]
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            features = features.cpu().numpy()

        # Store the path to patches, labels, and features
        patches_sorted.extend(patches)
        labels_sorted.extend(labels)
        features_sorted.extend(features)

    # Write the embeddings to the logs dir
    write_embeddings(logs_dir, patches_sorted, labels_sorted, features_sorted)

    # Activate tensorboard
    activate_tensorboard(logs_dir)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Spotlight")

    parser.add_argument("--model", type=str,
                        help="Path to Best Model and Weights File (.h5)")

    parser.add_argument('--patches', type=str,
                        help='The path to the patch labels csv file output the Patches tool')

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where results will be saved.")

    parser.add_argument('--project_folder', type=str, default="",
                        help='Path to existing projector project folder.')

    args = parser.parse_args()

    try:
        projector(args)
        print("Done.\n")

    except Exception as e:
        console_user(f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()