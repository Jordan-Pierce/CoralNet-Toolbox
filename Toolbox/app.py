import gradio as gr

import os
import subprocess

# ----------------------------------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------------------------------
def module_callback():
    """

    """
    subprocess.run(["python", r"C:\Users\jordan.pierce\Documents\GitHub\CoralNet-Toolbox\Toolbox\download_app.py"])

# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------

demo = gr.Blocks()

with demo:
    # Title
    gr.Markdown("# CoralNet Toolobox")

    # Input Parameters
    run_button = gr.Button("CoralNet Download")
    run = run_button.click(module_callback)


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, inbrowser=True)