import gradio as gr

import os
import sys
import time
import argparse
import traceback

from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import choose_files
from Toolbox.Pages.common import choose_directory
from Toolbox.Pages.common import SERVER_PORTS


from Toolbox.Tools.Common import DATA_DIR
from Toolbox.Tools.Common import LOG_PATH

from Toolbox.Tools.API import api

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(username, password, source_id_1, source_id_2, points, prefix, output_dir):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        username=username,
        password=password,
        source_id_1=source_id_1,
        source_id_2=source_id_2,
        points=points,
        prefix=prefix,
        output_dir=output_dir,
    )

    try:
        # Call the download function from your module
        api(args)
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

    with gr.Blocks(title="CoralNet API", analytics_enabled=False, theme=gr.themes.Soft()) as interface:
        # Title
        gr.Markdown("# CoralNet API")

        # Input Parameters
        with gr.Row():
            username = gr.Textbox(os.getenv('CORALNET_USERNAME'), label="Username", type='email')
            password = gr.Textbox(os.getenv('CORALNET_PASSWORD'), label="Password", type='password')

        with gr.Row():
            source_id_1 = gr.Textbox("", label="Source ID (for images)")
            source_id_2 = gr.Textbox("", label="Source ID (for model)")
            prefix = gr.Textbox("", label="Image Name Prefix")

        # Browse buttons
        points = gr.Textbox("", label="Selected Points File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_files, outputs=points, show_progress="hidden")

        # Browse buttons
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [username,
                                    password,
                                    source_id_1,
                                    source_id_2,
                                    points,
                                    prefix,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(check_interface)

        with gr.Accordion("Console Logs"):
            logs = gr.Textbox(label="")
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=SERVER_PORTS['api'], inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

interface = create_interface()

while True:
    time.sleep(0.5)
    if RESTART:
        exit_interface()