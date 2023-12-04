import gradio as gr

import json
import secrets
import numpy as np

import torch
import segmentation_models_pytorch as smp

from Toolbox.Pages.common import *

from Toolbox.Tools.Points import get_points
from Toolbox.Tools.Patches import crop_patch
from Toolbox.Tools.Classification import get_validation_augmentation

MODEL = None
COLOR_MAP = None

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
            print(f"NOTE: Loaded weights {model.name}")

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
        num_classes = len(class_map)

    else:
        raise Exception(f"ERROR: Class Map file provided doesn't exist.")

    global COLOR_MAP

    # Create color map for class categories
    if not COLOR_MAP:

        COLOR_MAP = {}

        # Generate random RGB values
        for idx in range(num_classes):
            rgb_values = [secrets.randbelow(256) for _ in range(3)]
            color = "#{:02x}{:02x}{:02x}".format(*rgb_values)
            COLOR_MAP[class_map[idx]] = color

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


def inference(patches, class_map):
    """

    """
    global MODEL

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get the preprocessing function that was used during training
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL.name, 'imagenet')

    # Convert patches to PyTorch tensor with validation augmentation and preprocessing
    validation_augmentation = get_validation_augmentation(height=224, width=224)

    patches = [torch.Tensor(preprocessing_fn(validation_augmentation(image=patch)['image'])) for patch in patches]
    patches_tensor = torch.stack(patches).permute(0, 3, 1, 2)
    patches_tensor = patches_tensor.to(device)

    with torch.no_grad():
        probabilities = MODEL(patches_tensor)

    # Convert PyTorch tensor to numpy array
    probabilities = probabilities.cpu().numpy()
    # Get predicted class indices
    predictions = np.argmax(probabilities, axis=1)
    class_predictions = np.array([class_map[v] for v in predictions]).astype(str)

    return probabilities, class_predictions


def create_annotations(image, x, y, class_predictions):
    """

    """
    # Buffer around predicted points
    radii = 20

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


def pipeline(model_path, class_map_path, image, sample_method, num_points, patch_size):
    """

    """
    global MODEL

    # Load the model if it isn't already
    if not MODEL:
        MODEL = load_model(model_path)

    # Open the class_map file
    class_map = get_class_map(class_map_path)

    # Get the data samples
    x, y, patches = prepare_data(image, patch_size, sample_method, num_points)

    # Perform inference
    probabilities, class_predictions = inference(patches, class_map)

    # Create the annotated image
    annotated_image = create_annotations(image, x, y, class_predictions)

    return annotated_image


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

        with gr.Group("Model"):
            #
            model_path = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model_path, show_progress="hidden")

            class_map_path = gr.Textbox(label="Selected Class Map File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=class_map_path, show_progress="hidden")

        with gr.Row():
            #
            sample_method = gr.Dropdown(label="Sample Method", multiselect=False,
                                        choices=['Uniform', 'Random', 'Stratified'])

            num_points = gr.Number(200, label="Number of Points", precision=0)

            patch_size = gr.Number(112, label="Patch Size", precision=0)

        with gr.Row():
            #
            image = gr.Image(label="Input Image", type='numpy')
            pred = gr.AnnotatedImage(label="Prediction",
                                     color_map=COLOR_MAP)

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run_button.click(pipeline,
                             [model_path,
                              class_map_path,
                              image,
                              sample_method,
                              num_points,
                              patch_size],
                             pred)

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