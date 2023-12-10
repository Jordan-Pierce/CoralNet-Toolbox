import os.path

import gradio as gr

import json
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize

import torch

from Toolbox.Pages.common import *

from Toolbox.Tools.SAM import get_sam_predictor
from Toolbox.Tools.SAM import get_exclusive_mask
from Toolbox.Tools.SAM import resize_image_aspect_ratio
from Toolbox.Tools.SAM import find_most_common_label_in_area

MODEL = None
EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def load_model(model_type, device, points_per_side, points_per_batch):
    """

    """
    try:
        # Load into the model
        model = get_sam_predictor(model_type, device, points_per_side, points_per_batch)
        gr.Info(f"Loaded weights for {model_type}")

        return model

    except Exception as e:
        raise Exception(f"ERROR: There was an issue loading the model\n{e}")


def inference(image, annotations, class_map):
    """

    """
    global MODEL

    # Empty cache
    torch.cuda.empty_cache()

    # Original image dimensions
    original_height, original_width = image.shape[0:2]

    # Resize the image to max width
    resized_image = resize_image_aspect_ratio(image)
    resized_height, resized_width = resized_image.shape[0:2]
    resized_area = resized_height * resized_width

    # Set the image in sam predictor
    masks = MODEL.generate(resized_image)
    # Sort based on area (larger first)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # Post processed semantic segmentation mask
    semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Post processed instance segmentation masks
    instance_masks = []

    # Loop through all masks generated
    for m_idx in range(len(masks)):

        # Get the generated mask
        resized_mask = masks[m_idx]['segmentation']

        # Check the area; if it's large, subtract it
        # from the other masks so there isn't overlap
        if masks[m_idx]['area'] / resized_area > 0.35:
            resized_mask = get_exclusive_mask(m_idx, masks)

        # Resize the mask back to original dimensions using nearest neighbor
        resized_mask = resized_mask.astype(np.uint8)
        mask = resize(resized_mask, (original_height, original_width), order=0)
        mask = mask.astype(bool)

        # Get the most common label in the mask area
        label = find_most_common_label_in_area(annotations, mask)

        # If it's not unlabeled
        if label:
            label = int(class_map[label])
            # Add to the semantic mask
            semantic_mask[mask] = label
            # Add to the instance masks
            instance_mask = np.zeros(shape=mask.shape, dtype=np.uint8)
            instance_mask[mask] = label
            instance_masks.append(instance_mask)

    return semantic_mask, instance_masks


def create_annotations(image, class_map, semantic_mask, instance_masks):
    """

    """
    semantic_annotations = []
    instance_annotations = []

    class_ids = list(class_map.values())
    class_names = list(class_map.keys())

    # Create the semantic annotations
    for idx in class_ids:
        # Get the current class
        class_id = class_ids[idx]
        class_name = class_names[idx]
        class_mask = (semantic_mask == class_id).copy()

        # Add to annotations
        semantic_annotations.append((class_mask, class_name))

    # Create the instance annotations
    for instance_mask in instance_masks:
        # Get the class
        class_id = instance_mask.max()
        class_name = class_names[class_id]

        # Add to annotations
        instance_annotations.append((instance_mask, class_name))

    return (image, semantic_annotations), (image, instance_annotations)


def pipeline(model_type, image_path, annotations_path, label_col, confidence, points_per_size, points_per_batch):
    """

    """
    global MODEL

    # Whether to run on GPU or CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the model
    MODEL = load_model(model_type, device, points_per_size, points_per_batch)

    # Read the image
    if os.path.exists(image_path):
        image_name = os.path.basename(image_path)
        image = imread(image_path)
    else:
        raise Exception("Image path provided does not exist")

    # Read the annotations
    if os.path.exists(annotations_path):
        annotations = pd.read_csv(annotations_path, index_col=0)
        annotations = annotations[annotations['Name'] == image_name]

        # Make sure image is in annotations
        if annotations.empty:
            gr.Warning("Image provided does not have any annotations")
            annotations = None

        # Get the points associated with label column, confidence
        if "suggestion" in label_col:
            annotations = annotations[annotations[label_col.replace("suggestion", "confidence")] >= confidence]

        # Make the subset of only necessary fields
        annotations = annotations[['Name', 'Row', 'Column', label_col]]
        annotations.columns = ['Name', 'Row', 'Column', 'Label']

        # Create a class mapping
        class_names = ["Unlabeled"] + sorted(annotations['Label'].unique().tolist())
        class_map = {l: i for i, l in enumerate(class_names)}

    else:
        raise Exception("Annotation path provided does not exist")

    try:
        # Perform inference
        gr.Info("Making predictions")
        semantic_mask, instance_masks = inference(image, annotations, class_map)

        # Create segmentation mask
        gr.Info("Creating visualization")
        semantic_annotations, instance_annotations = create_annotations(image,
                                                                        class_map,
                                                                        semantic_mask,
                                                                        instance_masks)

        return image, semantic_annotations, instance_annotations

    except Exception as e:
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

        with gr.Group("Annotations"):
            #
            annotations_path = gr.Textbox(label="Selected Annotation File")
            files_button = gr.Button("Browse Files")
            files_button.click(choose_file, outputs=annotations_path, show_progress="hidden")

            with gr.Row():
                #
                label_col = gr.Dropdown(label="Label Name Field", multiselect=False, allow_custom_value=True,
                                        choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

                confidence = gr.Slider(label="Point Confidence Filter", minimum=0, maximum=100, step=1)

        with gr.Group("SAM Model"):
            #
            with gr.Row():
                #
                model_type = gr.Dropdown(label="SAM Model Weights",
                                         choices=['vit_b', 'vit_l', 'vit_h'],
                                         multiselect=False, allow_custom_value=False)

                points_per_side = gr.Number(64, label="Number of Points (Squared)", precision=0)

                points_per_batch = gr.Number(128, label="Points per Batch (GPU dependent)", precision=0)

        # Input Image path
        image_path = gr.Textbox(label="Selected Image File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=image_path, show_progress="hidden")

        with gr.Row():
            # Input Image
            image = gr.Image(label="Input Image", type='numpy', interactive=False)

            # Output annotation, dataframe
            with gr.Tab("Semantic Output"):
                semantic_annotation = gr.AnnotatedImage(label="Prediction")

            with gr.Tab("Instance Output"):
                instance_annotation = gr.AnnotatedImage(label="Prediction")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run_button.click(pipeline,
                             # Inputs
                             [model_type,
                              image_path,
                              annotations_path,
                              label_col,
                              confidence,
                              points_per_side,
                              points_per_batch],
                             # Outputs
                             [image,
                              semantic_annotation,
                              instance_annotation])

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