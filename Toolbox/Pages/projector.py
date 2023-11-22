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
from Toolbox.Pages.common import choose_file
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import get_port

from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.Projector import projector

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(model, patches, output_dir, project_folder):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        model=model,
        patches=patches,
        output_dir=output_dir,
        project_folder=project_folder,
    )

    try:
        # Call the function
        projector(args)
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

    with gr.Blocks(title="Projector 📽️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Projector 📽️")

        with gr.Tab("New Project"):

            model = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model, show_progress="hidden")

            patches = gr.Textbox(label="Selected Patches File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=patches, show_progress="hidden")

            # Browse button
            output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Tab("Existing Project"):

            # Browse button
            project_folder = gr.Textbox(label="Existing Project Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=project_folder, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [model,
                                    patches,
                                    output_dir,
                                    project_folder])

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