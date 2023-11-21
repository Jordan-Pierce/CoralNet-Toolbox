import gradio as gr

import os
import sys
import time
import argparse
import traceback

from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import SERVER_PORTS

from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.Download import download
from Toolbox.Tools.Download import get_updated_labelset_list

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    # Convert source_ids to a list of strings
    source_ids_list = [str(id.strip()) for id in source_ids.split(",") if id.strip()]

    args = argparse.Namespace(
        username=username,
        password=password,
        source_ids=source_ids_list,
        source_df=source_df,
        labelset_df=labelset_df,
        sources_with=sources_with,
        output_dir=output_dir,
        headless=headless
    )

    try:
        # Call the download function from your module
        download(args)
        print("Done.")
    except Exception as e:
        return f"ERROR: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def set_interface_arguments():
    """

    """
    global RESTART
    RESTART = True


def create_interface():
    """

    """
    reset_logs()

    interface = gr.Blocks()

    with interface:
        # Title
        gr.Markdown("# CoralNet Downloader")

        # Input Parameters
        with gr.Tab("Download Source Data"):
            with gr.Row():
                username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
                password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

            with gr.Row():
                source_ids = gr.Textbox("4085", label="Source IDs (comma-separated)")
                headless = gr.Checkbox(label="Run Browser in Headless Mode", value=True)

        with gr.Tab("Download CoralNet Dataframes"):
            with gr.Row():
                source_df = gr.Checkbox(label="Download Source DataFrame")
                labelset_df = gr.Checkbox(label="Download Labelset DataFrame")

            sources_with = gr.Dropdown(label="Sources with Labelsets",
                                       choices=get_updated_labelset_list(),
                                       multiselect=True)

        # Browse buttons
        output_dir_path = gr.Textbox("", label="Selected Output Directory")
        output_dir_button = gr.Button("Browse Output Directory")
        output_dir_button.click(choose_directory, outputs=output_dir_path, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [username,
                                    password,
                                    source_ids,
                                    source_df,
                                    labelset_df,
                                    sources_with,
                                    output_dir_path,
                                    headless])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(set_interface_arguments)

        with gr.Accordion("Console Logs"):
            logs = gr.Textbox(label="")
            interface.load(read_logs, None, logs, every=1)

    interface.queue()
    interface.launch(share=False, server_port=SERVER_PORTS['download'], inbrowser=True)

    return interface


interface = create_interface()

while True:
    time.sleep(0.5)
    if RESTART:
        RESTART = False
        interface.close()
        interface = create_interface()