import gradio as gr

import cv2
import json
import numpy as np

import torch
import segmentation_models_pytorch as smp

from common import *

from Tools.Segmentation import get_validation_augmentation


MODEL = None
EXIT_APP = False
log_file = "segmentation_demo.log"


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
            color_mapping_dict = json.load(json_file)

        # Modify color map format
        class_names = list(color_mapping_dict.keys())
        class_ids = [color_mapping_dict[c]['id'] for c in class_names]
        class_colors = [color_mapping_dict[c]['color'] for c in class_names]

    else:
        raise Exception(f"ERROR: Class Map file provided doesn't exist.")

    return class_names, class_ids, class_colors


def inference(image):
    """

    """
    global MODEL

    # Empty cache
    torch.cuda.empty_cache()

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Original dimensions
    original_height, original_width = image.shape[0:2]

    # Resized dimensions
    height, width = 736, 1280

    # Get augmentation (resizing) and preprocessing functions
    augmentation = get_validation_augmentation(height, width)
    # Get the encoder name
    model_name = "-".join(MODEL.name.split("-")[1:])
    preprocessing_fn = smp.encoders.get_preprocessing_fn(model_name, 'imagenet')

    # Prepare the sample
    sample = augmentation(image=image)['image']
    sample = preprocessing_fn(sample)
    sample = torch.Tensor(sample).permute(2, 0, 1).unsqueeze(0)
    sample = sample.to(device)

    # Make prediction
    with torch.no_grad():
        pred = MODEL(sample)

    # Convert to numpy
    pred = pred.squeeze().cpu().numpy().round()
    pred = np.argmax(pred, axis=0)

    # Resize
    pred = cv2.resize(pred,
                      (original_width, original_height),
                      interpolation=cv2.INTER_NEAREST)

    return pred


def create_annotations(image, pred, class_names, class_ids):
    """

    """
    annotations = []

    for idx in class_ids:
        # Get the current class
        class_id = class_ids[idx]
        class_name = class_names[idx]
        class_mask = (pred == class_id).copy()

        # Add to annotations
        annotations.append((class_mask, class_name))

    return image, annotations


def pipeline(model_path, class_map_path, image):
    """

    """
    global MODEL

    # Load the model if it isn't already
    if not MODEL:
        MODEL = load_model(model_path)

    # Open the class_map file
    class_names, class_ids, class_colors = get_class_map(class_map_path)

    try:
        # Perform inference
        gr.Info("Making predictions")
        prediction = inference(image)

        # Create the annotated image
        annotated_image = create_annotations(image, prediction, class_names, class_ids)

        return annotated_image

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
    logger = Logger(log_file)
    logger.reset_logs()

    with gr.Blocks(title="Demo 🧙", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Demo 🧙")

        with gr.Group("Model"):
            #
            model_path = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model_path, show_progress="hidden")

            color_map_path = gr.Textbox(label="Selected Color Map File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=color_map_path, show_progress="hidden")

        with gr.Row():
            #
            image = gr.Image(label="Input Image", type='numpy')
            pred = gr.AnnotatedImage(label="Prediction")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run_button.click(pipeline,
                             [model_path,
                              color_map_path,
                              image],
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
    Logger(log_file).reset_logs()