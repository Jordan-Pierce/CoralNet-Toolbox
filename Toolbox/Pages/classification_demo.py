import os.path

import gradio as gr

import json
import numpy as np
import pandas as pd
from skimage.io import imread

import torch
import segmentation_models_pytorch as smp

from Toolbox.Pages.common import *

from Toolbox.Tools.Points import get_points
from Toolbox.Tools.Patches import crop_patch
from Toolbox.Tools.Classification import get_validation_augmentation

MODEL = None
EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def load_model(model_path):
    """

    """
    if os.path.exists(model_path):
        try:
            # Load into the model
            model = torch.load(model_path)
            gr.Info(f"Loaded weights for {model.name}")

            # Set the model to evaluation mode
            model.eval()

            return model

        except Exception as e:
            raise Exception(f"ERROR: There was an issue loading the model\n{e}")

    else:
        raise Exception("ERROR: Model provided doesn't exist.")


def get_class_map(class_map_path):
    """

    """
    if os.path.exists(class_map_path):
        with open(class_map_path, 'r') as json_file:
            class_mapping_dict = json.load(json_file)

        class_map = list(class_mapping_dict.keys())

    else:
        raise Exception(f"ERROR: Class Map file provided doesn't exist.")

    return class_map


def prepare_data(image, patch_size, sample_method, num_points):
    """

    """
    # Read it to get the size
    height, width = image.shape[0:2]

    # Slight offset
    min_width = 0 + 32
    max_width = width - 32
    min_height = 0 + 32
    max_height = height - 32

    # Get the points
    x, y = get_points(min_width, min_height, max_width, max_height, sample_method, num_points)

    # May be less if user chooses Uniform sampling
    num_points = len(list(zip(x, y)))

    patches = []

    # Get the patches
    for _ in range(num_points):
        patches.append(crop_patch(image, y[_], x[_], patch_size))

    patches = np.stack(patches)

    return x, y, patches


def inference(patches, class_map, batch_size):
    """

    """
    global MODEL

    # Empty cache
    torch.cuda.empty_cache()

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the preprocessing function that was used during training
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL.name, 'imagenet')

    # Convert patches to PyTorch tensor with validation augmentation and preprocessing
    validation_augmentation = get_validation_augmentation(height=224, width=224)

    # Initialize lists to store results
    all_probabilities = []
    all_class_predictions = []

    # Loop through patches in batches
    for i in range(0, len(patches), batch_size):
        # Get the current batch
        batch_patches = patches[i:i+batch_size]

        # Process and  convert batch patches to PyTorch tensor
        batch_patches = [preprocessing_fn(validation_augmentation(image=p)['image']) for p in batch_patches]
        batch_tensors = [torch.Tensor(p) for p in batch_patches]
        batch_tensor = torch.stack(batch_tensors).permute(0, 3, 1, 2)
        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            probabilities = MODEL(batch_tensor)

        # Convert PyTorch tensor to numpy array
        probabilities = probabilities.cpu().numpy()

        # Get predicted class indices
        predictions = np.argmax(probabilities, axis=1)
        class_predictions = np.array([class_map[v] for v in predictions]).astype(str)

        # Append results to lists
        all_probabilities.append(probabilities)
        all_class_predictions.append(class_predictions)

    # Concatenate results from all batches
    probabilities = np.concatenate(all_probabilities, axis=0)
    class_predictions = np.concatenate(all_class_predictions, axis=0)

    return probabilities, class_predictions


def create_annotations(image, x, y, class_predictions):
    """

    """
    # Buffer around predicted points
    radii = 15

    # Create a dictionary to store masks for each class
    class_masks = {label: np.zeros_like(image[:, :, 0]) for label in set(class_predictions)}

    for i in range(len(class_predictions)):
        current_x = x[i]
        current_y = y[i]
        current_class_label = class_predictions[i]

        # Ensure the coordinates are within the image dimensions
        if 0 <= current_y < image.shape[0] and 0 <= current_x < image.shape[1]:
            # Create a circular mask around the current (x, y) coordinates
            y_range, x_range = np.ogrid[-radii:radii + 1, -radii:radii + 1]
            circular_mask = (y_range ** 2 + x_range ** 2 <= radii ** 2)

            # Update the mask for the current class
            class_masks[current_class_label][current_y - radii:current_y + radii + 1,
            current_x - radii:current_x + radii + 1] += circular_mask

    # Convert the class_masks dictionary to a list of tuples
    annotations = [(mask, label) for label, mask in class_masks.items()]

    return image, annotations


def create_dataframe(image_name, x, y, class_predictions, output_dir):
    """

    """
    # Image name
    image_names = [image_name] * len(class_predictions)

    # Data
    data = list(zip(image_names, y, x, class_predictions))

    # Dataframe
    df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])

    # Output to disk
    output_dir = f"{output_dir}/demo"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{image_name.split('.')[0]}_{len(data)}_classification.csv")

    return df


def pipeline(model_path, class_map_path, image_path, sample_method, num_points, patch_size, batch_size, output_dir):
    """

    """
    global MODEL

    # Load the model if it isn't already
    if not MODEL:
        MODEL = load_model(model_path)

    # Open the class_map file
    class_map = get_class_map(class_map_path)

    # Read the image
    if os.path.exists(image_path):
        image_name = os.path.basename(image_path)
        image = imread(image_path)
    else:
        raise Exception("Image path provided does not exist")

    # Get the data samples
    gr.Info("Preparing samples")
    x, y, patches = prepare_data(image, patch_size, sample_method, num_points)

    try:
        # Perform inference
        gr.Info("Making predictions")
        probabilities, class_predictions = inference(patches, class_map, batch_size)

        # Create the annotated image
        gr.Info("Creating visualization")
        annotated_image = create_annotations(image, x, y, class_predictions)

        # Create the dataframe
        df = create_dataframe(image_name, x, y, class_predictions, output_dir)

        return image, annotated_image, df

    except:
        gr.Warning("GPU out of memory, try reducing patches and points")


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def exit_interface():
    """

    """
    global EXIT_APP
    EXIT_APP = True

    gr.Info("Please close the browser tab.")
    gr.Info("Stopped program successfully!")
    time.sleep(3)


def create_interface():
    """

    """
    Logger(LOG_PATH).reset_logs()

    with gr.Blocks(title="Demo 🧙", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Demo 🧙")

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Group("Model"):
            # Input Model
            model_path = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model_path, show_progress="hidden")

            class_map_path = gr.Textbox(label="Selected Class Map File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=class_map_path, show_progress="hidden")

        with gr.Row():
            # Parameters
            sample_method = gr.Dropdown(label="Sample Method", multiselect=False,
                                        choices=['Uniform', 'Random', 'Stratified'])

            num_points = gr.Number(200, label="Number of Points", precision=0)

            patch_size = gr.Number(112, label="Patch Size", precision=0)

            batch_size = gr.Number(512, label="Batch Size", precision=0)

        # Input Image path
        image_path = gr.Textbox(label="Selected Image File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=image_path, show_progress="hidden")

        with gr.Row():
            # Input Image
            image = gr.Image(label="Input Image", type='numpy', interactive=False)

            # Output annotation, dataframe
            with gr.Tab("Output Image"):
                annotated_image = gr.AnnotatedImage(label="Prediction")

            with gr.Tab("Output Dataframe"):
                df = gr.DataFrame(label="Prediction")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run_button.click(pipeline,
                             # Inputs
                             [model_path,
                              class_map_path,
                              image_path,
                              sample_method,
                              num_points,
                              patch_size,
                              batch_size,
                              output_dir],
                             # Outputs
                             [image,
                              annotated_image,
                              df])

            stop_button = gr.Button(value="Exit")
            stop = stop_button.click(exit_interface)

    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
interface = create_interface()

try:
    while True:
        time.sleep(0.5)
        if EXIT_APP:
            break
except:
    pass

finally:
    Logger(LOG_PATH).reset_logs()