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
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import SERVER_PORTS

from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.Download import download
from Toolbox.Tools.Download import get_updated_labelset_list

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_ids, source_df, labelset_df, sources_with, output_dir, headless):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    # Convert to a list of strings
    source_ids = [str(id.strip()) for id in source_ids.split(" ") if id.strip()]

    args = argparse.Namespace(
        username=username,
        password=password,
        source_ids=source_ids,
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

    with gr.Blocks(title="CoralNet Download", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# CoralNet Downloader")

        # Input Parameters
        with gr.Tab("Download Source Data"):
            with gr.Row():
                username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
                password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

            with gr.Row():
                source_ids = gr.Textbox("4085", label="Source IDs (space-separated)")
                headless = gr.Checkbox(label="Run Browser in Headless Mode", value=True)

        with gr.Tab("Download CoralNet Dataframes"):
            with gr.Row():
                source_df = gr.Checkbox(label="Download Source DataFrame")
                labelset_df = gr.Checkbox(label="Download Labelset DataFrame")

            sources_with = gr.Dropdown(label="Sources with Labelsets",
                                       choices=get_updated_labelset_list(),
                                       multiselect=True)

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

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
                                    output_dir,
                                    headless])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(check_interface)

        with gr.Accordion("Console Logs"):
            logs = gr.Textbox(label="")
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=SERVER_PORTS['download'], inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

interface = create_interface()

while True:
    time.sleep(0.5)
    if RESTART:
        exit_interface()