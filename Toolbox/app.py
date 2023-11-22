import gradio as gr

gr.Progress(track_tqdm=True)

import os
import sys
import time
import subprocess

from Toolbox.Pages.common import js
from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import get_port
from Toolbox.Pages.common import LOG_PATH
from Toolbox.Pages.common import PAGES_DIR

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------------------------------
def api_page():
    """

    """
    print("Opening API tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\api.py"])
    reset_logs()


def download_page():
    """

    """
    print("Opening Download tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\download.py"])
    reset_logs()


def labelset_page():
    """

    """
    print("Opening Labelset tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\labelset.py"])
    reset_logs()


def upload_page():
    """

    """
    print("Opening Upload tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\upload.py"])
    reset_logs()


def points_page():
    """

    """
    print("Opening Points tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\points.py"])
    reset_logs()


def patches_page():
    """

    """
    print("Opening Patches tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\patches.py"])
    reset_logs()


def projector_page():
    """

    """
    print("Opening Projector tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\projector.py"])
    reset_logs()


def visualize_page():
    """

    """
    print("Opening Visualize tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\visualize.py"])
    reset_logs()


def annotate_page():
    """

    """
    print("Opening Annotate tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\annotate.py"])
    reset_logs()


def classification_page():
    """

    """
    print("Opening Classification tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\classification.py"])
    reset_logs()


def img_inference_page():
    """

    """
    print("Opening Image Inference tool...")
    subprocess.run(["python", f"{PAGES_DIR}\img_inference.py"])
    reset_logs()


def classifier_demo_page():
    """

    """
    print("Opening Classifier Demo tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\classifier_demo.py"])
    reset_logs()


def sam_page():
    """

    """
    print("Opening SAM tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\sam.py"])
    reset_logs()


def sam_demo_page():
    """

    """
    print("Opening SAM Demo tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\sam_demo.py"])
    reset_logs()


def segmentation_page():
    """

    """
    print("Opening Segmentation tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\segmentation.py"])
    reset_logs()


def seg_inference_page():
    """

    """
    print("Opening Segmentation Inference tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\seg_inference.py"])
    reset_logs()


def segmentation_demo_page():
    """

    """
    print("Opening Segmentation Demo tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\segmentation_demo.py"])
    reset_logs()


def sfm_page():
    """

    """
    print("Opening SfM tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\sfm.py"])
    reset_logs()


def seg_3d_page():
    """

    """
    print("Opening Segmentation 3D tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\seg_3d.py"])
    reset_logs()


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def check_interface():
    """

    """
    global RESTART
    RESTART = True

    return


def exit_interface():
    """

    """
    reset_logs()

    print("")
    print("Stopped program successfully!")
    print("Connection closed!")
    print("")
    print("Please close the browser tab.")
    time.sleep(1)
    sys.exit(1)


def create_interface():
    # Reset logs function (assuming it is defined)
    reset_logs()

    with gr.Blocks(title="CoralNet Toolbox 🧰️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# CoralNet Toolbox 🧰"
                    "<p style='font-size:14px'>\nThe `CoralNet Toolbox` is an unofficial codebase that can be used to "
                    "augment processes associated with those on CoralNet. The following tools can assist with "
                    "interacting with CoralNet, and performing various other tasks related to analyzing images of "
                    "coral reefs and other benthic habits. These tools can perform processes programmatically via "
                    "command line, or through this GUI, and are meant to assist researchers and scientists working "
                    "with coral reefs, as well as for students and hobbyists interested in machine learning applied "
                    "to coral reefs. If you have any issues at all, please do not hesitate to reach out by [making a "
                    "ticket](https://github.com/Jordan-Pierce/CoralNet-Toolbox/issues) 🤦‍♂️</p>")

        gr.Markdown("## CoralNet Tools 🛠️ <p style='font-size:14px'>"
                    "\nUse these tool to interact with CoralNet</p>")
        with gr.Row():
            api_button = gr.Button("API 🕹️")
            api_process = api_button.click(api_page)

            download_button = gr.Button("Download ⬇️")
            download_process = download_button.click(download_page)

        with gr.Row():
            labelset_button = gr.Button("Labelset 📝")
            labelset_process = labelset_button.click(labelset_page)

            upload_button = gr.Button("Upload ⬆️️")
            upload_process = upload_button.click(upload_page)

        gr.Markdown("## Annotation Tools 🛠️ <p style='font-size:14px'>"
                    "\nUse these tool to create and visualize label data</p>")
        with gr.Row():
            points_button = gr.Button("Points 🏓")
            points_process = points_button.click(points_page)

            patches_button = gr.Button("Patches 🟩️️")
            patches_process = patches_button.click(patches_page)

        with gr.Row():
            projector_button = gr.Button("Projector 📽️")
            projector_process = projector_button.click(projector_page)

            visualize_button = gr.Button("Visualize 👓️")
            visualize_process = visualize_button.click(visualize_page)

        annotate_button = gr.Button("Annotate 🧮")
        annotate_process = annotate_button.click(annotate_page)

        gr.Markdown("## Image Classification Tools 🛠️ <p style='font-size:14px'>"
                    "\nUse these tool to train an image classifier</p>")
        with gr.Row():
            classification_button = gr.Button("Train 👨‍💻")
            classification_process = classification_button.click(classification_page)

            img_inference_button = gr.Button("Predict 🤖️")
            img_inference_button = img_inference_button.click(img_inference_page)

        with gr.Row():
            classifier_demo_button = gr.Button("Demo 🧙")
            classifier_demo_process = classifier_demo_button.click(classifier_demo_page)

        gr.Markdown("## Segment Anything Model 🛠️ <p style='font-size:14px'>"
                    "\nUse these tools to automatically create segmented images</p>")
        with gr.Row():
            sam_button = gr.Button("SAM 🧠")
            sam_process = api_button.click(sam_page)

            sam_demo_button = gr.Button("Demo 🧙")
            sam_demo_process = sam_demo_button.click(sam_demo_page)

        gr.Markdown("## Semantic Segmentation Tools 🛠️ <p style='font-size:14px'>"
                    "\nUse these tool to train an semantic segmentation model</p>")
        with gr.Row():
            segmentation_button = gr.Button("Train 👨‍💻")
            segmentation_process = segmentation_button.click(segmentation_page)

            seg_inference_button = gr.Button("Predict 🤖️")
            seg_inference_process = seg_inference_button.click(seg_inference_page)

        with gr.Row():
            segmentation_demo_button = gr.Button("Demo 🧙")
            segmentation_demo_process = segmentation_demo_button.click(segmentation_demo_page)

        gr.Markdown("## 3D Tools 🛠️ <p style='font-size:14px'>"
                    "\nUse these tool to train an image classifier</p>")
        with gr.Row():
            sfm_button = gr.Button("SfM (Metashape) 🧊")
            sfm_process = sfm_button.click(sfm_page)

            seg_3d_button = gr.Button("3D Semantic Segmentation 🤖️")
            seg_3d_process = seg_3d_button.click(seg_3d_page)

        gr.Markdown("## Shutdown 🥺 <p style='font-size:14px'>"
                    "\nTo properly close the application, press the button below</p>")
        with gr.Row():
            exit_button = gr.Button(value="Exit ⏏️")
            stop = exit_button.click(check_interface)

        # Console Logs
        with gr.Accordion("Console Logs"):
            logs = gr.Code(label="", language="shell", interactive=False, container=True)
            interface.load(read_logs, None, logs, every=1)

    # Launch the interface
    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
interface = create_interface()
sys.stdout = Logger(LOG_PATH)

while True:
    time.sleep(0.5)
    if RESTART:
        exit_interface()
