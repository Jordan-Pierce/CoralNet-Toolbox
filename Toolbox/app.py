import gradio as gr

import subprocess

from Toolbox.Pages.common import SERVER_PORTS
from Toolbox.Pages.common import PAGES_DIR


# ----------------------------------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------------------------------
def download_page():
    """

    """
    subprocess.run(["python", f"{PAGES_DIR}\\download.py"])


def test_page():
    """

    """
    subprocess.run(["python", f"{PAGES_DIR}\\test.py"])


# ----------------------------------------------------------------------------------------------------------------------
# Main Gradio Interface
# ----------------------------------------------------------------------------------------------------------------------

demo = gr.Blocks()

with demo:
    # Title
    gr.Markdown("# CoralNet Toolbox")

    # Input Parameters
    api_button = gr.Button("CoralNet API")
    api_process = api_button.click(download_page)

    # Input Parameters
    download_button = gr.Button("CoralNet Download")
    download_process = download_button.click(download_page)

    # Input Parameters
    test_button = gr.Button("Gradio Test")
    test_process = test_button.click(test_page)

if __name__ == "__main__":
    demo.launch(share=False, server_port=SERVER_PORTS['main'], inbrowser=True)