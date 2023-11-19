import os
import sys
import argparse
import warnings
import traceback
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread

import tensorflow as tf
from tensorboard.plugins import projector as P

keras = tf.keras
from keras.models import load_model

from Common import log
from Common import get_now
from Common import print_progress

from Classification import precision
from Classification import recall
from Classification import f1_score

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------
def load_patches(paths):
    """

    """
    patches = []

    for path in paths:

        if not os.path.exists(path):
            log(f"WARNING: {path} does not exist, skipping.")

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
    metadata_file = f"{logs_dir}\\metadata.csv"
    tensor_file = f"{logs_dir}\\features.tsv"
    sprite_file = f"{logs_dir}\\sprite.jpg"

    # Write the metadata
    with open(os.path.join(logs_dir, metadata_file), "w") as f:
        for label in labels:
            f.write("{}\n".format(label))

    # Write the configs
    with open(os.path.join(logs_dir, tensor_file), "w") as f:
        for tensor in features:
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
    log("NOTE: Activating Tensorboard...")
    tensorboard_exe = os.path.join(os.path.dirname(sys.executable), 'Scripts', 'tensorboard')
    process = subprocess.Popen([tensorboard_exe, "--logdir", logs_dir])

    try:
        while True:
            continue

    except Exception:
        log("NOTE: Deactivating Tensorboard...")

    finally:
        process.terminate()


def projector(args):
    """

    """

    log("\n###############################################")
    log("Projector")
    log("###############################################\n")

    # Check that the user has GPU available
    if tf.config.list_physical_devices('GPU'):
        log("NOTE: Found GPU")
        gpus = tf.config.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        log("WARNING: No GPU found; defaulting to CPU")

    # ---------------------------------------------------------------------------------------
    # If there's an existing project
    if args.project_folder:
        try:
            if os.path.exists(args.project_folder):
                log("NOTE: Opening existing project")
                activate_tensorboard(args.project_folder)
            else:
                log("ERROR: Project file provided does not exist, check provided input")
            raise Exception
        except Exception as e:
            sys.exit(1)

    # Source directory setup
    logs_dir = f"{args.output_dir}\\projector\\{get_now()}"
    os.makedirs(logs_dir)

    # Get the patches
    if os.path.exists(args.patches):
        # Patch dataframe
        patches_df = pd.read_csv(args.patches, index_col=0)
        patches_df = patches_df.dropna()
        patches_df.replace({'NotCoral': 'Substrate', 'DVR': 'Background', 'SclBar': 'Color Palette'}, inplace=True)
        # Get the image base names
        assert "Path" in patches_df.columns, log(f"ERROR: 'Path' not in provided csv")
        assert "Name" in patches_df.columns, log("ERROR: 'Image Name' not in provided csv")
        image_names = patches_df['Image Name'].unique().tolist()
    else:
        raise Exception(f"ERROR: Patches dataframe {args.patches} does not exist")
    pass

    # Model Weights
    if os.path.exists(args.model):
        try:
            # Load the model with custom metrics
            custom_objects = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
            model = load_model(args.model, custom_objects=custom_objects)
            feature_extractor = model.layers[0]
            log(f"NOTE: Loaded model {args.model}")

        except Exception as e:
            log(f"ERROR: There was an issue loading the model\n{e}")
            sys.exit(1)
    else:
        log("ERROR: Model provided doesn't exist.")
        sys.exit(1)

    # Loop through each of the images, extract features from associated patches
    patches_sorted = []
    labels_sorted = []
    features_sorted = []

    log("NOTE: Creating feature embeddings")

    for i_idx, image_name in enumerate(image_names):
        # Patches for current image
        current_patches = patches_df[patches_df['Image Name'] == image_name]
        # Get the patch labels
        labels = current_patches['Label'].values.tolist()
        # Get the patch arrays
        patches = load_patches(current_patches['Path'].values.tolist())
        # Get the features for patches
        features = feature_extractor.predict(np.stack(patches), verbose=0)

        # Store the path to patches, labels, and features
        patches_sorted.extend(patches)
        labels_sorted.extend(labels)
        features_sorted.extend(features)

        # Gooey
        print_progress(i_idx + 1, len(image_names))

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

    parser = argparse.ArgumentParser(description="Projector")

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
        log("Done.\n")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()