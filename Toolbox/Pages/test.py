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


# ----------------------------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------------------------


def module_callback(value):
    """

    """
    sys.stdout = Logger(LOG_PATH)

    try:
        # Call function
        for val in range(int(value)):
            print(val)
            time.sleep(1)

        print("Done.")
    except Exception as e:
        return f"ERROR: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------

demo = gr.Blocks()

with demo:
    # Title
    gr.Markdown("# Gradio Test")

    # Input Parameters
    value = gr.Textbox(label="Value", placeholder="100")

    with gr.Row():
        # Run button (callback)
        run_button = gr.Button("Run")
        run = run_button.click(module_callback,
                               [value])

        stop_button = gr.Button(value="Stop")
        stop = stop_button.click(fn=None, inputs=None, outputs=None, cancels=[run])

    with gr.Accordion("Console Logs"):
        logs = gr.Textbox(label="")
        demo.load(read_logs, None, logs, every=1)

if __name__ == "__main__":
    reset_logs()
    demo.launch(share=False, server_port=SERVER_PORTS['test'], inbrowser=True)