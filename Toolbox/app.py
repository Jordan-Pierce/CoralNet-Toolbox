import gradio as gr

import os
import sys
import time
import subprocess

from Toolbox.Pages.common import Logger
from Toolbox.Pages.common import read_logs
from Toolbox.Pages.common import reset_logs
from Toolbox.Pages.common import LOG_PATH
from Toolbox.Pages.common import SERVER_PORTS
from Toolbox.Pages.common import PAGES_DIR

RESTART = False


# ----------------------------------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------------------------------
def download_page():
    """

    """
    print("Opening Download tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\download.py"])
    reset_logs()


def api_page():
    """

    """
    print("Opening API tool...")
    subprocess.run(["python", f"{PAGES_DIR}\\api.py"])
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
    """

    """
    reset_logs()

    with gr.Blocks(title="CoralNet Toolbox", analytics_enabled=False, theme=gr.themes.Soft()) as interface:
        # Title
        gr.Markdown("# CoralNet Toolbox")

        api_button = gr.Button("CoralNet API")
        api_process = api_button.click(api_page)

        download_button = gr.Button("CoralNet Download")
        download_process = download_button.click(download_page)

        stop_button = gr.Button(value="Exit")
        stop = stop_button.click(check_interface)

        with gr.Accordion("Console Logs"):
            logs = gr.Textbox(label="")
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=SERVER_PORTS['main'], inbrowser=True, show_error=True)

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