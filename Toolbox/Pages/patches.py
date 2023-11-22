import gradio as gr

import os
import sys
import time
import argparse
import traceback

from Toolbox.Pages.common import js
from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import choose_files
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import get_port


from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.Patches import patches

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(image_dir, annotation_file, image_column, label_column, patch_size, output_dir):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        image_dir=image_dir,
        annotation_file=annotation_file,
        image_column=image_column,
        label_column=label_column,
        patch_size=patch_size,
        output_dir=output_dir,
    )

    try:
        # Call the function
        patches(args)
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}\n{traceback.format_exc()}")


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
    """

    """
    reset_logs()

    with gr.Blocks(title="Patches 🟩", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Patches 🟩")

        # Browse button
        image_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Image Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=image_dir, show_progress="hidden")

        annotation_file = gr.Textbox(label="Selected Annotation File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_files, outputs=annotation_file, show_progress="hidden")

        with gr.Row():
            image_column = gr.Textbox("Name", label="Image Name Field")

            label_column = gr.Dropdown(label="Label Name Field", multiselect=False,
                                       choices=['Label'] + [f'Machine suggestion {n + 1}' for n in range(5)])

            patch_size = gr.Number(112, label="Patch Size", precision=0)

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [image_dir,
                                    annotation_file,
                                    image_column,
                                    label_column,
                                    patch_size,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(check_interface)

        with gr.Accordion("Console Logs"):
            logs = gr.Code(label="", language="shell", interactive=False, container=True)
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

interface = create_interface()

while True:
    time.sleep(0.5)
    if RESTART:
        exit_interface()